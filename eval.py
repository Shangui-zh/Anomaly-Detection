"""
Evaluation Scripts
"""
from __future__ import absolute_import
from __future__ import division
from collections import namedtuple, OrderedDict
from network import mynn
import argparse
import logging
import os
import torch
import time
import numpy as np

from config import cfg, assert_and_infer_cfg
import network
import optimizer
from ood_metrics import fpr_at_95_tpr
from tqdm import tqdm

from PIL import Image
from sklearn.metrics import roc_auc_score, roc_curve, auc, precision_recall_curve, average_precision_score
import torchvision.transforms as standard_transforms
import datasets
import torchvision
import cv2
import pdb

from utils.img_utils import Compose, Normalize, ToTensor
from network.deepv3 import BoundarySuppressionWithSmoothing

import random

dirname = os.path.dirname(__file__)
pretrained_model_path = os.path.join(dirname, 'pretrained/r101_best_model.pth')
   

# Argument Parser
parser = argparse.ArgumentParser(description='Semantic Segmentation')
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--arch', type=str, default='network.deepv3.DeepR101V3PlusD_OS8',
                    help='Network architecture.')
parser.add_argument('--dataset', type=str, default='cityscapes',
                    help='possible datasets for statistics; cityscapes')
parser.add_argument('--fp16', action='store_true', default=False,
                    help='Use Nvidia Apex AMP')
parser.add_argument('--local_rank', default=0, type=int,
                    help='parameter used by apex library')
parser.add_argument('--trunk', type=str, default='resnet101',
                    help='trunk model, can be: resnet101 (default), resnet50')
parser.add_argument('--bs_mult', type=int, default=2,
                    help='Batch size for training per gpu')
parser.add_argument('--bs_mult_val', type=int, default=1,
                    help='Batch size for Validation per gpu')
parser.add_argument('--class_uniform_pct', type=float, default=0,
                    help='What fraction of images is uniformly sampled')
parser.add_argument('--class_uniform_tile', type=int, default=1024,
                    help='tile size for class uniform sampling')
parser.add_argument('--batch_weighting', action='store_true', default=False,
                    help='Batch weighting for class (use nll class weighting using batch stats')
parser.add_argument('--jointwtborder', action='store_true', default=False,
                    help='Enable boundary label relaxation')

parser.add_argument('--snapshot', type=str, default=pretrained_model_path)
parser.add_argument('--restore_optimizer', action='store_true', default=False)

parser.add_argument('--date', type=str, default='default',
                    help='experiment directory date name')
parser.add_argument('--exp', type=str, default='default',
                    help='experiment directory name')
parser.add_argument('--tb_tag', type=str, default='',
                    help='add tag to tb dir')
parser.add_argument('--ckpt', type=str, default='logs/ckpt',
                    help='Save Checkpoint Point')
parser.add_argument('--tb_path', type=str, default='logs/tb',
                    help='Save Tensorboard Path')
parser.add_argument('--syncbn', action='store_true', default=True,
                    help='Use Synchronized BN')
parser.add_argument('--dist_url', default='tcp://127.0.0.1:', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--backbone_lr', type=float, default=0.0,
                    help='different learning rate on backbone network')
parser.add_argument('--pooling', type=str, default='mean',
                    help='pooling methods, average is better than max')

parser.add_argument('--ood_dataset_path', type=str,
                    default='/sda1/yijun/Static',
                    help='OoD dataset path')

# Anomaly score mode - msp, max_logit, standardized_max_logit
parser.add_argument('--score_mode', type=str, default='standardized_max_logit',
                    help='score mode for anomaly [msp, max_logit, standardized_max_logit, Ours]')

# Boundary suppression configs
parser.add_argument('--enable_boundary_suppression', type=bool, default=True,
                    help='enable boundary suppression')
parser.add_argument('--boundary_width', type=int, default=4,
                    help='initial boundary suppression width')
parser.add_argument('--boundary_iteration', type=int, default=4,
                    help='the number of boundary iterations')

# Dilated smoothing configs
parser.add_argument('--enable_dilated_smoothing', type=bool, default=True,
                    help='enable dilated smoothing')
parser.add_argument('--smoothing_kernel_size', type=int, default=7,
                    help='kernel size of dilated smoothing')
parser.add_argument('--smoothing_kernel_dilation', type=int, default=6,
                    help='kernel dilation rate of dilated smoothing')

parser.add_argument('--my_epoch', type=str, default='epoch_74',
                    help='our epoch')

parser.add_argument('--my_score_mode', type=str, default='maxLogit',
                    help='score mode for anomaly [msp, maxLogit, SML, maxLogit_ours, SML_ours]')
parser.add_argument('--my_city', type=str, default='strasbourg',
                    help='city file')
parser.add_argument('--ood_data', type=str, default='Static',
                    help='Static, LAF, RA')

args = parser.parse_args()


# Enable CUDNN Benchmarking optimization
#torch.backends.cudnn.benchmark = True
random_seed = cfg.RANDOM_SEED
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(random_seed)

args.world_size = 1

print(f'World Size: {args.world_size}')
if 'WORLD_SIZE' in os.environ:
    # args.apex = int(os.environ['WORLD_SIZE']) > 1
    args.world_size = int(os.environ['WORLD_SIZE'])
    print("Total world size: ", int(os.environ['WORLD_SIZE']))

torch.cuda.set_device(args.local_rank)
print('My Rank:', args.local_rank)
# Initialize distributed communication
args.dist_url = args.dist_url + str(8000 + (int(time.time()%1000))//10)

torch.distributed.init_process_group(backend='nccl',
                                     init_method=args.dist_url,
                                     world_size=args.world_size,
                                     rank=args.local_rank)

def get_net():
    """
    Main Function
    """
    # Set up the Arguments, Tensorboard Writer, Dataloader, Loss Fn, Optimizer
    assert_and_infer_cfg(args)

    net = network.get_net(args, criterion=None, criterion_aux=None)

    net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net)
    net = network.warp_network_in_dataparallel(net, args.local_rank)

    if args.snapshot:
        epoch, mean_iu = optimizer.load_weights(net, None, None,
                            args.snapshot, args.restore_optimizer)
        print(f"Loading completed. Epoch {epoch} and mIoU {mean_iu}")
    else:
        raise ValueError(f"snapshot argument is not set!")

    
    class_mean = np.load(f'stats/SML/cityscapes_SML_origin_label_mean.npy', allow_pickle=True)
    class_var = np.load(f'stats/SML/cityscapes_SML_origin_label_var.npy', allow_pickle=True)
    
    net.module.set_statistics(mean=class_mean.item(), var=class_var.item())

    torch.cuda.empty_cache()
    net.eval()

    return net

def preprocess_image(x, mean_std):
    x = Image.fromarray(x)
    x = standard_transforms.ToTensor()(x)
    x = standard_transforms.Normalize(*mean_std)(x)

    x = x.cuda()

    if len(x.shape) == 3:
        x = x.unsqueeze(0)
    return x


if __name__ == '__main__':
    image_root_path = './input'
    if not os.path.exists(image_root_path):
        raise ValueError(f"Dataset directory {image_root_path} doesn't exist!")
    mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    anomaly_score_list = []

    net = get_net()
    
    if args.my_score_mode == 'SML':
        class_mean = np.load(f'stats/SML/cityscapes_SML_origin_label_mean.npy', allow_pickle=True).item()
        class_var = np.load(f'stats/SML/cityscapes_SML_origin_label_var.npy', allow_pickle=True).item()
    else:
        class_mean = np.load(f'stats/ours/cityscapes_ours_mean.npy', allow_pickle=True).item()
        class_var = np.load(f'stats/ours/cityscapes_ours_var.npy', allow_pickle=True).item()
        class_max = np.load(f'stats/ours/cityscapes_ours_max.npy', allow_pickle=True).item()
        class_min = np.load(f'stats/ours/cityscapes_ours_min.npy', allow_pickle=True).item()

    multi_scale = BoundarySuppressionWithSmoothing(
                    boundary_suppression=args.enable_boundary_suppression,
                    boundary_width=args.boundary_width,
                    boundary_iteration=args.boundary_iteration,
                    dilated_smoothing=args.enable_dilated_smoothing,
                    kernel_size=args.smoothing_kernel_size,
                    dilation=args.smoothing_kernel_dilation)
    multi_scale.cuda()
    
    city_path = image_root_path
    cities = os.listdir(city_path)
    for image_file in tqdm(cities):
        image_path = os.path.join(city_path, image_file)
        find_number = image_file[:4]

        # 3 x H x W
        image = np.array(Image.open(image_path).convert('RGB')).astype('uint8')

        with torch.no_grad():
            
            image = preprocess_image(image, mean_std)
            main_out, anomaly_score = net(image)
            maxLogit, prediction = main_out.detach().max(1)
            anomaly_score = maxLogit
            
            if args.my_score_mode == 'maxLogit':
                anomaly_score = maxLogit
            elif args.my_score_mode == 'MSP':
                anomaly_score, _ = nn.Softmax(dim=1)(main_out.detach()).max(1)
            elif args.my_score_mode == 'SML':
                for c in range(datasets.num_classes):
                    anomaly_score = torch.where(prediction == c,
                                            (anomaly_score - class_mean[c]) / np.sqrt(class_var[c]),
                                            anomaly_score)
                anomaly_score = multi_scale(anomaly_score, prediction.cuda())
            elif args.my_score_mode == 'Ours':
                for c in range(datasets.num_classes):
                    if not np.isnan(class_min[c]) and class_min[c]!=0 : 
                        anomaly_score = torch.where(prediction == c,
                                                (anomaly_score - (class_mean[c]+class_max[c]-class_min[c])/(2)) / class_var[c],
                                                anomaly_score)
                anomaly_score = multi_scale(anomaly_score, prediction.cuda())
            np.save(f'./output/'+find_number+'_anomaly.npy', anomaly_score.squeeze(0).cpu().numpy())
            np.save(f'./output/'+find_number+'_segmentation.npy', prediction.squeeze(0).cpu().numpy())

        del main_out
