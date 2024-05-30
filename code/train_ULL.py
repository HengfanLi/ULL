import os
import sys
from tqdm import tqdm
from tensorboardX import SummaryWriter
import shutil
import argparse
import logging
import time
import random
import numpy as np
import torch
import torch.optim as optim
from torchvision import transforms
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from networks.vnet import VNet
from networks.ResNet34 import Resnet34
from utils import ramps, losses
from dataloaders.la_heart import LAHeart, RandomCrop, ToTensor, TwoStreamBatchSampler
import skimage.segmentation 
from scipy.ndimage import distance_transform_edt as distance
parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='../data/2018LA_Seg_Training Set/', help='Name of Experiment')               # todo change dataset path
parser.add_argument('--exp', type=str,  default="ULL37", help='model_name')                               # todo model name
parser.add_argument('--max_iterations', type=int,  default=6000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=4, help='batch_size per gpu')
parser.add_argument('--labeled_bs', type=int, default=2, help='labeled_batch_size per gpu')
parser.add_argument('--base_lr', type=float,  default=0.01, help='maximum epoch number to train')
parser.add_argument('--deterministic', type=int,  default=1, help='whether use deterministic training')
parser.add_argument('--seed', type=int,  default=337, help='random seed')
parser.add_argument('--gpu', type=str,  default='0', help='GPU to use')
### costs
parser.add_argument('--ema_decay', type=float,  default=0.999, help='ema_decay')
parser.add_argument('--consistency_type', type=str,  default="mse", help='consistency_type')
parser.add_argument('--consistency', type=float,  default=0.1, help='consistency')
parser.add_argument('--consistency_rampup', type=float,  default=40.0, help='consistency_rampup')
parser.add_argument('--temperature', type=float, default=0.1, help='temperature of sharpening')
args = parser.parse_args()

train_data_path = args.root_path
snapshot_path = "../model/" + args.exp + "/"

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
batch_size = args.batch_size * len(args.gpu.split(','))
max_iterations = args.max_iterations
base_lr = args.base_lr
labeled_bs = args.labeled_bs

if args.deterministic:
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

num_classes = 2
patch_size = (112, 112, 80)
T = 0.1
Good_student = 0 # 0: vnet 1:resnet
def sharpening(P):
    T = 1/args.temperature
    P_sharpen = P ** T / (P ** T + (1-P) ** T)
    return P_sharpen
def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)

def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)

def worker_init_fn(worker_id):
    random.seed(args.seed+worker_id)

def gateher_two_patch(vec):
    b, c, num = vec.shape
    cat_result = []
    for i in range(c-1):
        temp_line = vec[:,i,:].unsqueeze(1)  # b 1 c
        star_index = i+1
        rep_num = c-star_index
        repeat_line = temp_line.repeat(1, rep_num,1)
        two_patch = vec[:,star_index:,:]
        temp_cat = torch.cat((repeat_line,two_patch),dim=2)
        cat_result.append(temp_cat)

    result = torch.cat(cat_result,dim=1)
    return  result


if __name__ == "__main__":
    ## make logger file
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    if os.path.exists(snapshot_path + '/code'):
        shutil.rmtree(snapshot_path + '/code')
    shutil.copytree('.', snapshot_path + '/code', shutil.ignore_patterns(['.git','__pycache__']))

    logging.basicConfig(filename=snapshot_path+"/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    def create_model(name ='vnet'):
        # Network definition
        if name == 'vnet':
            net = VNet(n_channels=1, n_classes=num_classes, normalization='batchnorm', has_dropout=True)
            model = net.cuda()
        if name == 'resnet34':
            net = Resnet34(n_channels=1, n_classes=num_classes, normalization='batchnorm', has_dropout=True)
            model = net.cuda()
        return model

    model_vnet = create_model(name='vnet')
    model_resnet = create_model(name='resnet34')

    db_train = LAHeart(base_dir=train_data_path,
                               split='train',
                               train_flod='train0.list',                   # todo change training flod
                               common_transform=transforms.Compose([
                                   RandomCrop(patch_size),
                               ]),
                               sp_transform=transforms.Compose([
                                   ToTensor(),
                               ]))

    labeled_idxs = list(range(16))           # todo set labeled num
    unlabeled_idxs = list(range(16, 80))     # todo set labeled num all_sample_num

    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, batch_size, batch_size - labeled_bs)
    trainloader = DataLoader(db_train, batch_sampler=batch_sampler, num_workers=4, pin_memory=True,
                             worker_init_fn=worker_init_fn)
    vnet_optimizer = optim.SGD(model_vnet.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    resnet_optimizer = optim.SGD(model_resnet.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)

    if args.consistency_type == 'mse':
        consistency_criterion = losses.softmax_mse_loss
    elif args.consistency_type == 'kl':
        consistency_criterion = losses.softmax_kl_loss
    else:
        assert False, args.consistency_type

    writer = SummaryWriter(snapshot_path+'/log')
    logging.info("{} itertations per epoch".format(len(trainloader)))
    iter_num = 0
    max_epoch = max_iterations//len(trainloader)+1
    lr_ = base_lr
    model_vnet.train()
    model_resnet.train()

    for epoch_num in tqdm(range(max_epoch), ncols=70):
        time1 = time.time()
        for i_batch, sampled_batch in enumerate(trainloader):
            time2 = time.time()
            print('epoch:{},i_batch:{}'.format(epoch_num,i_batch))
            volume_batch1, volume_label1 = sampled_batch[0]['image'], sampled_batch[0]['label']
            volume_batch2, volume_label2 = sampled_batch[1]['image'], sampled_batch[1]['label']

            v_input,v_label = volume_batch1.cuda(), volume_label1.cuda()
            r_input,r_label = volume_batch2.cuda(), volume_label2.cuda()

            v_outputs = model_vnet(v_input)
            r_outputs = model_resnet(r_input)

            ## calculate the supervised loss
            v_loss_seg = F.cross_entropy(v_outputs[:labeled_bs], v_label[:labeled_bs])
            # print('v_loss_seg',v_loss_seg)
            v_outputs_soft = F.softmax(v_outputs, dim=1)
            v_loss_seg_dice = losses.dice_loss(v_outputs_soft[:labeled_bs, 1, :, :, :], v_label[:labeled_bs] == 1)

            r_loss_seg = F.cross_entropy(r_outputs[:labeled_bs], r_label[:labeled_bs])
            r_outputs_soft = F.softmax(r_outputs, dim=1)
            r_loss_seg_dice = losses.dice_loss(r_outputs_soft[:labeled_bs, 1, :, :, :], r_label[:labeled_bs] == 1)           
                       
 ####################CRE
            v_outputs_soft2 = F.softmax(v_outputs, dim=1)
            r_outputs_soft2 = F.softmax(r_outputs, dim=1)

            la_v_predict = torch.max(v_outputs_soft2[:labeled_bs, :, :, :, :], 1, )[1]
            la_r_predict = torch.max(r_outputs_soft2[:labeled_bs, :, :, :, :], 1, )[1]
            threshold = (0.75+0.25*ramps.sigmoid_rampup(iter_num, max_iterations))*np.log(2)
            #diff_mask = (la_v_predict!=la_r_predict).float()
            with torch.no_grad():
                gt_v_boundary_mask = compute_foreground_boundary_mask(v_label[:labeled_bs].cpu(
                ).numpy(), v_outputs[:labeled_bs, ...].shape)
                gt_v_boundary_mask   =torch.tensor(gt_v_boundary_mask).cuda()
                gt_r_boundary_mask = compute_foreground_boundary_mask(r_label[:labeled_bs].cpu(
                ).numpy(), r_outputs[:labeled_bs, 0, ...].shape)
                gt_r_boundary_mask   =torch.tensor(gt_r_boundary_mask).cuda()


            v_entropy = -1*torch.sum(v_outputs_soft2[:labeled_bs]*torch.log(v_outputs_soft2[:labeled_bs]+1e-6), dim=1)
            r_entropy = -1*torch.sum(r_outputs_soft2[:labeled_bs]*torch.log(r_outputs_soft2[:labeled_bs]+1e-6), dim=1)

            # diff_mask = (((la_v_predict==la_r_predict)&(la_v_predict!=v_label[:labeled_bs]))\
            #              |((v_entropy<r_entropy)&(la_v_predict!=v_label[:labeled_bs])&(la_r_predict==r_label[:labeled_bs]))\
            #              |((v_entropy>r_entropy)&(la_r_predict!=r_label[:labeled_bs])&(la_v_predict==v_label[:labeled_bs]))).float()
            diff_mask = ((((la_v_predict==1)&(la_r_predict==1))&(la_v_predict!=v_label[:labeled_bs]))\
                        |(((la_v_predict==0)&(la_r_predict==0))&(la_v_predict!=v_label[:labeled_bs]))\
                         |((v_entropy<r_entropy)&(la_v_predict!=v_label[:labeled_bs])&(la_r_predict==r_label[:labeled_bs]))\
                         |((v_entropy>r_entropy)&(la_r_predict!=r_label[:labeled_bs])&(la_v_predict==v_label[:labeled_bs]))\
                         |(abs(v_entropy-r_entropy)>0.3)
                         ).float()

       
            v_hard_entropy = (v_entropy*diff_mask).float()
            v_hard_entropy  =v_hard_entropy[v_hard_entropy>0]
            print('v_hard_entropy ',torch.unique(v_hard_entropy))
            v_hard_mean_entropy = torch.mean(v_hard_entropy)
            r_hard_entropy = (r_entropy*diff_mask).float()
            r_hard_entropy  =r_hard_entropy[r_hard_entropy>0]
            r_hard_mean_entropy = torch.mean(r_hard_entropy)
     
           # total_hard_entropy =  torch.cat((v_hard_entropy, r_hard_entropy), dim=0)
           # print(' total_hard_entropy: ', total_hard_entropy)
            mean_total_hard_entropy= (v_hard_mean_entropy+r_hard_mean_entropy)/2
            #v_mean_total_hard_entropy = v_mean_total_hard_entropy.item()
           # print('mean_total_hard_entropy: ',mean_total_hard_entropy)
           # print('diff_mask: ',diff_mask)
            diff_v_soft = (v_outputs_soft2*gt_v_boundary_mask*diff_mask).float()
            diff_r_soft = (r_outputs_soft2*gt_r_boundary_mask*diff_mask).float()
   
            diff_v_entropy = -1*torch.sum(diff_v_soft[:labeled_bs]*torch.log(diff_v_soft[:labeled_bs]+1e-6), dim=1)
            print('diff_v_entropy: ',torch.unique(diff_v_entropy[diff_v_entropy>0]))
            diff_r_entropy = -1*torch.sum(diff_r_soft[:labeled_bs]*torch.log(diff_r_soft[:labeled_bs]+1e-6), dim=1)
            total_entropy =  torch.cat((diff_v_entropy, diff_r_entropy), dim=0)
            print('total_entropy: ',torch.unique(total_entropy[total_entropy>0]))

            
            v_boundary_entropy = v_entropy*diff_mask
            r_boundary_entropy = r_entropy*diff_mask
            # print('v_bounary: ',torch.unique(v_boundary_entropy))
            # print('r_bounary: ',torch.unique(r_boundary_entropy))
            # v_boundary_entropy=-1*torch.sum(v_boundary_soft*torch.log(v_boundary_soft+1e-6), dim=1)
            # r_boundary_entropy=-1*torch.sum(r_boundary_soft*torch.log(r_boundary_soft+1e-6), dim=1)
            # print('v_bounary: ',torch.unique(v_boundary_entropy))
            # print('r_bounary: ',torch.unique(r_boundary_entropy))
            ##########################

            # total_entropy =  torch.cat((v_boundary_entropy, r_boundary_entropy), dim=0)
            # print('total_entropy: ',torch.unique(total_entropy))
    
            # non_zero_values = total_entropy[total_entropy > 0]
            # total_entropy_non_zero_count = torch.count_nonzero(non_zero_values)
            # print('non_zero_values.shape: ',non_zero_values.shape)
            # print('non_zero_values: ',torch.unique(non_zero_values))
            
            # no_zero = total_entropy>0
            # total_entropy = total_entropy[no_zero]
            # mean_entropy = torch.mean(non_zero_values)
            # std_entropy = torch.std(non_zero_values)
            # print('mean_entropy ' ,mean_entropy)
            # print('std_entropy ',std_entropy)
            v_non_zero_values=v_boundary_entropy[v_boundary_entropy > 0]
            r_non_zero_values=r_boundary_entropy[r_boundary_entropy > 0]
            v_mean_entropy= torch.median(v_non_zero_values)
            r_mean_entropy= torch.median(r_non_zero_values)
           # v_std_entropy = torch.std(v_non_zero_values)
           
            mean_entropy = (v_mean_entropy+r_mean_entropy)/2
            # min_entropy = torch.min(total_entropy)
            # min_entropy =min_entropy.item()
            

            v_hard_region_mask = (v_entropy>mean_entropy).float()
            r_hard_region_mask = (r_entropy>mean_entropy).float()

            v_hard_region_Loss=losses.hard_region_Loss(v_outputs[:labeled_bs],v_label[:labeled_bs],v_hard_region_mask)
            #print('v_hard_region_loss: ',v_hard_region_Loss)
            r_hard_region_Loss=losses.hard_region_Loss(r_outputs[:labeled_bs],r_label[:labeled_bs],r_hard_region_mask)
            v_mse_dist = consistency_criterion(v_outputs_soft2[:labeled_bs, 1, :, :, :], v_label[:labeled_bs] )
            r_mse_dist = consistency_criterion(r_outputs_soft2[:labeled_bs, 1, :, :, :], r_label[:labeled_bs] )
            v_mse      = torch.sum(diff_mask * v_mse_dist) / (torch.sum(diff_mask) + 1e-16)
            r_mse      = torch.sum(diff_mask * r_mse_dist) / (torch.sum(diff_mask) + 1e-16)
            print('v_mse: ',v_mse)
            v_supervised_loss =  (v_loss_seg + v_loss_seg_dice) + 0.4 * v_mse
            r_supervised_loss =  (r_loss_seg + r_loss_seg_dice) + 0.4 * r_mse
############unsup
 
            v_outputs_clone = v_outputs_soft[labeled_bs:, :, :, :, :].clone().detach()
            r_outputs_clone = r_outputs_soft[labeled_bs:, :, :, :, :].clone().detach()
            v_outputs_clone1 = torch.pow(v_outputs_clone, 1 / T)
            r_outputs_clone1 = torch.pow(r_outputs_clone, 1 / T)
            v_outputs_clone2 = torch.sum(v_outputs_clone1, dim=1, keepdim=True)
            r_outputs_clone2 = torch.sum(r_outputs_clone1, dim=1, keepdim=True)
            v_outputs_PLable = torch.div(v_outputs_clone1, v_outputs_clone2)
            r_outputs_PLable = torch.div(r_outputs_clone1, r_outputs_clone2)
            un_v_boundary_entropy = -1*torch.sum(v_outputs_soft2[labeled_bs:]*torch.log(v_outputs_soft2[labeled_bs:]+1e-6), dim=1)
            un_r_boundary_entropy = -1*torch.sum(r_outputs_soft2[labeled_bs:]*torch.log(r_outputs_soft2[labeled_bs:]+1e-6), dim=1)
            un_total_entropy=  torch.cat((un_v_boundary_entropy, un_r_boundary_entropy), dim=0)
            un_total_entropy = un_total_entropy.view(-1)
            un_total_mean_entropy = torch.mean(un_total_entropy)
            un_total_std_entropy = torch.std(un_total_entropy)
            un_v_hard_region_mask = (un_v_boundary_entropy>(mean_entropy)).float()
            un_r_hard_region_mask = (un_r_boundary_entropy>(mean_entropy)).float()
            # v_fir_condition=(un_v_boundary_entropy> mean_entropy) &(un_r_boundary_entropy< mean_entropy) 
            # r_fir_condition=(un_r_boundary_entropy> mean_entropy) &(un_v_boundary_entropy< mean_entropy)
            
            #print('threshold ',threshold)
            v_fir_shar_pseudo_mask = ((un_v_boundary_entropy> (threshold)) & \
                                         (un_r_boundary_entropy< threshold)).float()  
            r_fir_shar_pseudo_mask = ((un_v_boundary_entropy< (threshold)) & \
                                         (un_r_boundary_entropy> threshold)).float()        
            v_fir_shar_dist=consistency_criterion(r_outputs_soft[labeled_bs:, :, :, :, :].detach(),v_outputs_soft[labeled_bs:, :, :, :, :])
            r_fir_shar_dist=consistency_criterion(v_outputs_soft[labeled_bs:, :, :, :, :].detach(),r_outputs_soft[labeled_bs:, :, :, :, :])
            b, c, w, h, d = v_fir_shar_dist.shape
            
            v_fir_loss = torch.sum( v_fir_shar_pseudo_mask * v_fir_shar_dist) / (torch.sum( v_fir_shar_pseudo_mask) + 1e-16)
            r_fir_loss = torch.sum( r_fir_shar_pseudo_mask * r_fir_shar_dist) / (torch.sum( r_fir_shar_pseudo_mask) + 1e-16)
            print('v_fir_loss: ',v_fir_loss)
            v_predict = torch.max(v_outputs_soft2[labeled_bs:, :, :, :, :], 1, )[1]
            r_predict = torch.max(v_outputs_soft2[labeled_bs:, :, :, :, :], 1, )[1]
            
            sec_shar_pseudo_mask = ((un_v_boundary_entropy<(0.20)) & \
                                    (un_r_boundary_entropy<(0.20))& \
                                        (((v_predict==1)&(r_predict==1))|((v_predict==0)&(r_predict==0)))
                                        ).float()

            un_v_good_mask = (un_v_boundary_entropy<un_r_boundary_entropy).float()
            un_r_good_mask = (un_v_boundary_entropy>un_r_boundary_entropy).float()                

            
            v_sec_dist = consistency_criterion(r_outputs_PLable,v_outputs_soft[labeled_bs:, :, :, :, :])
            r_sec_dist = consistency_criterion(v_outputs_PLable,r_outputs_soft[labeled_bs:, :, :, :, :])            
            sec_dist1 = consistency_criterion(v_outputs_soft[labeled_bs:, :, :, :, :],r_outputs_soft[labeled_bs:, :, :, :, :])
            sec_dist2 = consistency_criterion(r_outputs_soft[labeled_bs:, :, :, :, :],v_outputs_soft[labeled_bs:, :, :, :, :])
            
            sec_loss1 = torch.sum(sec_shar_pseudo_mask*sec_dist1) / (torch.sum( sec_shar_pseudo_mask) + 1e-16)
            sec_loss2 = torch.sum(sec_shar_pseudo_mask*sec_dist2) / (torch.sum( sec_shar_pseudo_mask) + 1e-16)
            sec_loss = sec_loss1+sec_loss2

          #thir_shar_pseudo_mask =  ((un_v_boundary_entropy>(0.2)) & (un_r_boundary_entropy>(0.2))  ).float()
            un_v_good_mask2 = ((un_v_boundary_entropy>(0.2)) & (un_r_boundary_entropy>(0.2)) &((un_r_boundary_entropy-un_v_boundary_entropy)>0.05)).float()
            un_r_good_mask2 = ((un_v_boundary_entropy>(0.2)) & (un_r_boundary_entropy>(0.2)) &((un_v_boundary_entropy-un_r_boundary_entropy)>0.05)).float()
            v_thi_dist = consistency_criterion(r_outputs_PLable,v_outputs_soft[labeled_bs:, :, :, :, :].clone())
            r_thi_dist = consistency_criterion(v_outputs_PLable,r_outputs_soft[labeled_bs:, :, :, :, :].clone())
            v_thir_loss = torch.sum(un_r_good_mask2*v_thi_dist) / (torch.sum( un_r_good_mask2) + 1e-16)
            r_thir_loss = torch.sum(un_v_good_mask2*r_thi_dist) / (torch.sum( un_v_good_mask2) + 1e-16)
            print('v_thir_loss ',v_thir_loss)
            v_un_shar_loss  = v_thir_loss+r_thir_loss+sec_loss
            r_un_shar_loss  = r_fir_loss+r_thir_loss#r_sec_loss#+#+r_sec_loss#+thir_loss
           # print('v_un_shar_loss ',v_un_shar_loss)
         #############uncertainty minimize   
            
            
            # um_loss = losses.entropy_loss(outputs_avg_soft, C=2)
            # print('um_loss: ',um_loss)
            consistency_weight = get_current_consistency_weight(iter_num//150)
            v_loss = v_supervised_loss+consistency_weight*v_un_shar_loss+r_supervised_loss + consistency_weight*r_un_shar_loss#+um_loss*0.1# + v_un_shar_loss*consistency_weight 
            #r_loss = r_supervised_loss + consistency_weight*r_un_shar_loss #+um_loss
            #v_loss = v_supervised_loss + v_un_shar_loss*consistency_weight 
            #r_loss = r_supervised_loss + r_un_shar_loss*consistency_weight 
            
            vnet_optimizer.zero_grad()
            resnet_optimizer.zero_grad()
            v_loss.backward()
            #r_loss.backward()
            vnet_optimizer.step()
            resnet_optimizer.step()
            writer.add_scalar('lr', lr_, iter_num)
            writer.add_scalar('loss/v_loss', v_loss, iter_num)
            writer.add_scalar('loss/v_loss_seg', v_loss_seg, iter_num)
            writer.add_scalar('loss/v_loss_seg_dice', v_loss_seg_dice, iter_num)
            writer.add_scalar('loss/v_supervised_loss', v_supervised_loss, iter_num)
            writer.add_scalar('loss/v_mse', v_mse, iter_num)
            #writer.add_scalar('loss/r_loss', r_loss, iter_num)
            writer.add_scalar('loss/r_loss_seg', r_loss_seg, iter_num)
            writer.add_scalar('loss/r_loss_seg_dice', r_loss_seg_dice, iter_num)
            writer.add_scalar('loss/r_supervised_loss', r_supervised_loss, iter_num)
            # writer.add_scalar('loss/r_mse', r_mse, iter_num)
            # writer.add_scalar('train/Good_student', Good_student, iter_num)
            writer.add_scalar('loss/v_fir_loss', v_fir_loss, iter_num)
            writer.add_scalar('loss/r_fir_loss', r_fir_loss, iter_num)
           # writer.add_scalar('loss/v_sec_loss', v_sec_loss, iter_num)
           # writer.add_scalar('loss/r_sec_loss', r_sec_loss, iter_num)
            logging.info(
                'iteration ： %d v_supervised_loss : %f v_loss_seg : %f v_loss_seg_dice : %f  r_supervised_loss : %f r_loss_seg : %f r_loss_seg_dice : %f v_fir_loss: %f r_fir_loss: %f \
                 mean_entropy : %f , un_total_mean_entropy: %f'  %
                (iter_num,
                 v_supervised_loss.item(), v_loss_seg.item(), v_loss_seg_dice.item(), 
                 r_supervised_loss.item(), r_loss_seg.item(), r_loss_seg_dice.item(),v_fir_loss.item(),r_fir_loss.item(),
                 mean_entropy,un_total_mean_entropy
                 ))
            if iter_num % 1000 == 0 and iter_num!= 0:
                save_mode_path_vnet = os.path.join(snapshot_path, 'vnet_iter_' + str(iter_num) + '.pth')
                torch.save(model_vnet.state_dict(), save_mode_path_vnet)
                logging.info("save model to {}".format(save_mode_path_vnet))

                save_mode_path_resnet = os.path.join(snapshot_path, 'resnet_iter_' + str(iter_num) + '.pth')
                torch.save(model_resnet.state_dict(), save_mode_path_resnet)
                logging.info("save model to {}".format(save_mode_path_resnet))
            ## change lr
            if iter_num % 2500 == 0 and iter_num!= 0:
                lr_ = lr_ * 0.1
                for param_group in vnet_optimizer.param_groups:
                    param_group['lr'] = lr_
                for param_group in resnet_optimizer.param_groups:
                    param_group['lr'] = lr_

            if iter_num >= max_iterations:
                break
            time1 = time.time()

            iter_num = iter_num + 1
            if iter_num >= max_iterations:
                break
        if iter_num >= max_iterations:
            break

    save_mode_path_vnet = os.path.join(snapshot_path, 'vnet_iter_' + str(max_iterations) + '.pth')
    torch.save(model_vnet.state_dict(), save_mode_path_vnet)
    logging.info("save model to {}".format(save_mode_path_vnet))

    save_mode_path_resnet = os.path.join(snapshot_path, 'resnet_iter_' + str(max_iterations) + '.pth')
    torch.save(model_resnet.state_dict(), save_mode_path_resnet)
    logging.info("save model to {}".format(save_mode_path_resnet))

    writer.close()
