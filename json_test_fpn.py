# --------------------------------------------------------
# Pytorch FPN
# Licensed under The MIT License [see LICENSE for details]
# Written by Jianwei Yang, based on code from faster R-CNN
# --------------------------------------------------------
# python train_fpn.py --dataset pascal_voc --net res101 --bs 2 --nw 2 --lr 0.01 --lr_decay_step 8 --cuda
# python json_fpn_cap.py --net res101 --checksession 1 --checkepoch 4 --checkpoint 2968 --cuda --load_dir models --image_dir /media/gong/d16d2182-cf80-41fa-8c1a-ce10d644c955/tianchi/data/chongqing1_round1_testA_20191223/images

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import cv2
import _init_paths
import os
import sys
import numpy as np
import argparse
import pprint
import pdb
import time
import logging

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
from model.rpn.bbox_transform import clip_boxes
import torchvision.transforms as transforms
from torch.utils.data.sampler import Sampler
from model.rpn.bbox_transform import bbox_transform_inv
from roi_data_layer.roidb import combined_roidb
from roi_data_layer.roibatchLoader import roibatchLoader
from model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from model.utils.net_utils import weights_normal_init, save_net, load_net, \
      adjust_learning_rate, save_checkpoint, clip_gradient
from scipy.misc import imread
from model.fpn.resnet import resnet
import pdb
from model.utils.blob import im_list_to_blob
from model.roi_layers import nms
from model.utils.net_utils import save_net, load_net, vis_detections
import json
try:
    xrange          # Python 2
except NameError:
    xrange = range  # Python 3


categoryList={'bg':0,'cap break':1,'cap deform':2,'cap edge defect':3,'cap spin':4,'cap spot':5,'label lean':6,'label flatten':7,'label bubble':8,'code norm':9,'code error':10}

def parse_args():
  """
  Parse input arguments
  """
  parser = argparse.ArgumentParser(description='TEST A FPN NETWORK')
  parser.add_argument('--dataset', dest='dataset',
                      help='training dataset',
                      default='pascal_voc', type=str)
  parser.add_argument('--net', dest='net',
                    help='res101, res152, etc',
                    default='res101', type=str)
  parser.add_argument('--start_epoch', dest='start_epoch',
                      help='starting epoch',
                      default=1, type=int)
  parser.add_argument('--epochs', dest='max_epochs',
                      help='number of epochs to train',
                      default=50, type=int)
  parser.add_argument('--disp_interval', dest='disp_interval',
                      help='number of iterations to display',
                      default=100, type=int)
  parser.add_argument('--checkpoint_interval', dest='checkpoint_interval',
                      help='number of iterations to display',
                      default=10000, type=int)

  parser.add_argument('--save_dir', dest='save_dir',
                      help='directory to save models', default="models",
                      nargs=argparse.REMAINDER)
  parser.add_argument('--nw', dest='num_workers',
                      help='number of worker to load data',
                      default=0, type=int)
  parser.add_argument('--cuda', dest='cuda',default=1,
                      help='whether use CUDA',
                      action='store_true')
  parser.add_argument('--mGPUs', dest='mGPUs',
                      help='whether use multiple GPUs',
                      action='store_true')
  parser.add_argument('--lscale', dest='lscale',
                      help='whether use large scale',
                      action='store_true')
  parser.add_argument('--bs', dest='batch_size',
                      help='batch_size',
                      default=1, type=int)
  parser.add_argument('--cag', dest='class_agnostic',
                      help='whether perform class_agnostic bbox regression',
                      action='store_true')

  parser.add_argument('--load_dir', dest='load_dir',
                      help='directory to load models',
                      default="models")
  parser.add_argument('--image_dir', dest='image_dir',
                      help='directory to load images for demo',
                      default="/media/gong/d16d2182-cf80-41fa-8c1a-ce10d644c955/tianchi/data/chongqing1_round1_testA_20191223/images")
# log and diaplay
  parser.add_argument('--use_tfboard', dest='use_tfboard',
                      help='whether use tensorflow tensorboard',
                      default=False, type=bool)
  parser.add_argument('--checksession', dest='checksession',
                      help='checksession to load model',
                      default=1, type=int)
  parser.add_argument('--checkepoch', dest='checkepoch',
                      help='checkepoch to load network',
                      default=49, type=int)
  parser.add_argument('--checkpoint', dest='checkpoint',
                      help='checkpoint to load network',
                      default=2968, type=int)
  parser.add_argument('--vis', dest='vis',
                      help='visualization mode',
                      action='store_true')
  parser.add_argument('--webcam_num', dest='webcam_num',
                      help='webcam ID number',
                      default=-1, type=int)
  parser.add_argument('--set', dest='set_cfgs',
                      help='set config keys', default=None,
                      nargs=argparse.REMAINDER)
  args = parser.parse_args()
  return args


class sampler(Sampler):
  def __init__(self, train_size, batch_size):
    num_data = train_size
    self.num_per_batch = int(num_data / batch_size)
    self.batch_size = batch_size
    self.range = torch.arange(0,batch_size).view(1, batch_size).long()
    self.leftover_flag = False
    if num_data % batch_size:
      self.leftover = torch.arange(self.num_per_batch*batch_size, num_data).long()
      self.leftover_flag = True
  def __iter__(self):
    rand_num = torch.randperm(self.num_per_batch).view(-1,1) * self.batch_size
    self.rand_num = rand_num.expand(self.num_per_batch, self.batch_size) + self.range

    self.rand_num_view = self.rand_num.view(-1)

    if self.leftover_flag:
      self.rand_num_view = torch.cat((self.rand_num_view, self.leftover),0)

    return iter(self.rand_num_view)

  def __len__(self):
    return num_data

def _print(str, logger):
  print(str)
  logger.info(str)

def _get_image_blob(im):
  """Converts an image into a network input.
  Arguments:
    im (ndarray): a color image in BGR order
  Returns:
    blob (ndarray): a data blob holding an image pyramid
    im_scale_factors (list): list of image scales (relative to im) used
      in the image pyramid
  """
  im_orig = im.astype(np.float32, copy=True)
  im_orig -= cfg.PIXEL_MEANS

  im_shape = im_orig.shape
  im_size_min = np.min(im_shape[0:2])
  im_size_max = np.max(im_shape[0:2])

  processed_ims = []
  im_scale_factors = []

  for target_size in cfg.TEST.SCALES:
    im_scale = float(target_size) / float(im_size_min)
    # Prevent the biggest axis from being more than MAX_SIZE
    if np.round(im_scale * im_size_max) > cfg.TEST.MAX_SIZE:
      im_scale = float(cfg.TEST.MAX_SIZE) / float(im_size_max)
    im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale,
            interpolation=cv2.INTER_LINEAR)
    im_scale_factors.append(im_scale)
    processed_ims.append(im)

  # Create a blob to hold the input images
  blob = im_list_to_blob(processed_ims)

  return blob, np.array(im_scale_factors)

if __name__ == '__main__':

  args = parse_args()

  print('Called with args:')
  print(args)
  args.cfg_file = "cfgs/{}_ls.yml".format(args.net) if args.lscale else "cfgs/{}.yml".format(args.net)

  if args.cfg_file is not None:
    cfg_from_file(args.cfg_file)
  if args.set_cfgs is not None:
    cfg_from_list(args.set_cfgs)

  cfg.USE_GPU_NMS = args.cuda
  cfg.TRAIN.USE_FLIPPED = False
  print('Using config:')
  pprint.pprint(cfg)
  np.random.seed(cfg.RNG_SEED)

  input_dir = args.load_dir + "/" + args.net + "/" + args.dataset
  if not os.path.exists(input_dir):
    raise Exception('There is no input directory for loading network from ' + input_dir)
  load_name = os.path.join(input_dir,
    'fpn_{}_{}_{}.pth'.format(args.checksession, args.checkepoch, args.checkpoint))

  pascal_classes = np.asarray(['__background__',
                      'cap break','cap deform','cap edge defect','cap spin','cap spot','code norm','code error'])

  if args.net == 'res101':
    FPN = resnet(pascal_classes, 101, pretrained=False, class_agnostic=args.class_agnostic)
  elif args.net == 'res50':
    FPN = resnet(pascal_classes, 50, pretrained=False, class_agnostic=args.class_agnostic)
  elif args.net == 'res152':
    FPN = resnet(pascal_classes, 152, pretrained=False, class_agnostic=args.class_agnostic)
  else:
    print("network is not defined")
    pdb.set_trace()

  FPN.create_architecture()


  print("load checkpoint %s" % (load_name))
  checkpoint = torch.load(load_name)
  FPN.load_state_dict(checkpoint['model'])
  if 'pooling_mode' in checkpoint.keys():
    cfg.POOLING_MODE = checkpoint['pooling_mode']
    _print("loaded checkpoint %s" % (load_name), logging)


  print('load model successfully!')
# initilize the tensor holder here.
  im_data = torch.FloatTensor(1)
  im_info = torch.FloatTensor(1)
  num_boxes = torch.LongTensor(1)
  gt_boxes = torch.FloatTensor(1)

  # ship to cuda
  if args.cuda > 0:
    im_data = im_data.cuda()
    im_info = im_info.cuda()
    num_boxes = num_boxes.cuda()
    gt_boxes = gt_boxes.cuda()

  # make variable
  im_data = Variable(im_data, volatile=True)
  im_info = Variable(im_info, volatile=True)
  num_boxes = Variable(num_boxes, volatile=True)
  gt_boxes = Variable(gt_boxes, volatile=True)

  if args.cuda > 0:
    cfg.CUDA = True
    
  if args.mGPUs:
    FPN = nn.DataParallel(FPN)

  if args.cuda:
    FPN.cuda()

  FPN.eval()

  # FPN.train()

  start = time.time()
  max_per_image = 100
  thresh = 0.05
  vis = True

  article_info = {}
  json_data = json.loads(json.dumps(article_info))

  data_images=[]
  data_annotations=[]

  json_file = open('cap.json','w',encoding='utf-8')

  webcam_num = args.webcam_num
  # Set up webcam or get image directories
  if webcam_num >= 0 :
    cap = cv2.VideoCapture(webcam_num)
    num_images = 0
  else:
    imglist = os.listdir(args.image_dir)
    num_images = len(imglist)

  print('Loaded Photo: {} images.'.format(num_images))

  while (num_images >= 0):
      total_tic = time.time()
      if webcam_num == -1:
        num_images -= 1

      # Get image from the webcam
      if webcam_num >= 0:
        if not cap.isOpened():
          raise RuntimeError("Webcam could not open. Please check connection.")
        ret, frame = cap.read()
        im_in = np.array(frame)
      # Load the demo image
      else:
        im_file = os.path.join(args.image_dir, imglist[num_images])
        # im = cv2.imread(im_file)
        im_in = np.array(imread(im_file))
      if len(im_in.shape) == 2:
        im_in = im_in[:,:,np.newaxis]
        im_in = np.concatenate((im_in,im_in,im_in), axis=2)
      # rgb -> bgr
      im = im_in[:,:,::-1]


      if im_in.shape[0]>1000:
        continue
      img_id=int(imglist[num_images].split("img_")[1].split(".")[0])
      print(img_id)
      single_image={}
      single_image["file_name"]=imglist[num_images]
      single_image["id"]=img_id
      data_images.append(single_image)
      blobs, im_scales = _get_image_blob(im)
      assert len(im_scales) == 1, "Only single-image batch implemented"
      im_blob = blobs
      im_info_np = np.array([[im_blob.shape[1], im_blob.shape[2], im_scales[0]]], dtype=np.float32)

      im_data_pt = torch.from_numpy(im_blob)
      im_data_pt = im_data_pt.permute(0, 3, 1, 2)
      im_info_pt = torch.from_numpy(im_info_np)
      print(im_data.shape)
      with torch.no_grad():
              im_data.resize_(im_data_pt.size()).copy_(im_data_pt)
              im_info.resize_(im_info_pt.size()).copy_(im_info_pt)
              gt_boxes.resize_(1, 1, 5).zero_()
              num_boxes.resize_(1).zero_()

      # pdb.set_trace()
      det_tic = time.time()

      # rois, cls_prob, bbox_pred, rpn_loss_cls, rpn_loss_box, \
      #          RCNN_loss_cls, RCNN_loss_bbox, \
      #          rois_label = FPN(im_data, im_info, gt_boxes, num_boxes)

      rois, cls_prob, bbox_pred, \
          _, _, _, _, _ =FPN(im_data, im_info, gt_boxes, num_boxes)
      scores = cls_prob.data
      boxes = rois.data[:, :, 1:5]

      # print(scores)
      # print(bbox_pred)
      if cfg.TEST.BBOX_REG:
          # Apply bounding-box regression deltas
          box_deltas = bbox_pred.data
          if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
          # Optionally normalize targets by a precomputed mean and stdev
            if args.class_agnostic:
                if args.cuda > 0:
                    box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                               + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                else:
                    box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS) \
                               + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS)

                box_deltas = box_deltas.view(1, -1, 4)
            else:
                if args.cuda > 0:
                    box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                               + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                else:
                    box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS) \
                               + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS)
                box_deltas = box_deltas.view(1, -1, 4 * len(pascal_classes))

          pred_boxes = bbox_transform_inv(boxes, box_deltas, 1)
          pred_boxes = clip_boxes(pred_boxes, im_info.data, 1)
      else:
          # Simply repeat the boxes, once for each class
          pred_boxes = np.tile(boxes, (1, scores.shape[1]))

      pred_boxes /= im_scales[0]

      scores = scores.squeeze()
      pred_boxes = pred_boxes.squeeze()
      det_toc = time.time()
      detect_time = det_toc - det_tic
      misc_tic = time.time()
      if vis:
          im2show = np.copy(im)

      bg_flg=0

      for j in xrange(1, len(pascal_classes)):
          inds = torch.nonzero(scores[:,j]>thresh).view(-1)
          # if there is det
          if inds.numel() > 0:
            cls_scores = scores[:,j][inds]
            _, order = torch.sort(cls_scores, 0, True)
            if args.class_agnostic:
              cls_boxes = pred_boxes[inds, :]
            else:
              cls_boxes = pred_boxes[inds][:, j * 4:(j + 1) * 4]
            
            cls_dets = torch.cat((cls_boxes, cls_scores.unsqueeze(1)), 1)
            # cls_dets = torch.cat((cls_boxes, cls_scores), 1)
            cls_dets = cls_dets[order]
            # keep = nms(cls_dets, cfg.TEST.NMS, force_cpu=not cfg.USE_GPU_NMS)
            keep = nms(cls_boxes[order, :], cls_scores[order], cfg.TEST.NMS)
            cls_dets = cls_dets[keep.view(-1).long()]
#             print(pascal_classes[j],cls_dets.cpu())
            dets_np=cls_dets.cpu().numpy()
            for i in range(np.minimum(10, dets_np.shape[0])):
              x,y,x_max,y_max = dets_np[i, :4]
              w=x_max-x
              h=y_max-y
              score = dets_np[i, -1]
              if score > 0.2:
                print(pascal_classes[j],score,x,y,w,h)
                single_anno={}
                single_anno["image_id"]=img_id
                single_anno["bbox"]=[float(x),float(y),float(w),float(h)]
                single_anno["category_id"]=categoryList[pascal_classes[j]]
                single_anno["score"]=float(score)
                data_annotations.append(single_anno)
                bg_flg=1

            if vis:
              im2show = vis_detections(im2show, pascal_classes[j], cls_dets.cpu().numpy(), 0.2)

      if(bg_flg==0):
        print("This is bg image!")
        single_anno={}
        single_anno["image_id"]=img_id
        single_anno["bbox"]=[0,0,100,100]
        single_anno["category_id"]=0
        single_anno["score"]=1.0
        data_annotations.append(single_anno)


      misc_toc = time.time()
      nms_time = misc_toc - misc_tic

      if webcam_num == -1:
          sys.stdout.write('im_detect: {:d}/{:d} {:.3f}s {:.3f}s   \r' \
                           .format(num_images + 1, len(imglist), detect_time, nms_time))
          sys.stdout.flush()

      if vis and webcam_num == -1:
          # cv2.imshow('test', im2show)
          # cv2.waitKey(0)
          result_path = os.path.join("images", imglist[num_images][:-4] + "_det.jpg")
          cv2.imwrite(result_path, im2show)

  json_data['images']=data_images
  json_data['annotations']=data_annotations
  article = json.dump(json_data,json_file, ensure_ascii=False)