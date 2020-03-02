# --------------------------------------------------------
# Pytorch FPN
# Licensed under The MIT License [see LICENSE for details]
# Written by Jianwei Yang, based on code from faster R-CNN
# --------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
import numpy as np
import argparse
import pprint
import pdb
import time

import torch
from torch.autograd import Variable
import torch.nn as nn

from torch.utils.data.sampler import Sampler
from roi_data_layer.roidb import combined_roidb
from roi_data_layer.roibatchLoader import roibatchLoader
from model.utils.config import cfg, cfg_from_file, cfg_from_list
from model.utils.net_utils import adjust_learning_rate, save_checkpoint
from model.fpn_cascade.resnet import resnet
# from model.fpn.non_cascade.detnet_backbone import detnet as detnet_noncascade
from tensorboardX import SummaryWriter
from model.utils.summary import *
import pdb

try:
    xrange  # Python 2
except NameError:
    xrange = range  # Python 3


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
    # parser.add_argument('exp_name', type=str, default=None, help='experiment name')
    parser.add_argument('--dataset', dest='dataset',
                        help='training dataset',
                        default='pascal_voc', type=str)
    parser.add_argument('--net', dest='net',
                        help='detnet59, etc',
                        default='res101', type=str)
    parser.add_argument('--start_epoch', dest='start_epoch',
                        help='starting epoch',
                        default=1, type=int)
    parser.add_argument('--epochs', dest='max_epochs',
                        help='number of epochs to train',
                        default=20, type=int)
    parser.add_argument('--disp_interval', dest='disp_interval',
                        help='number of iterations to display',
                        default=100, type=int)
    parser.add_argument('--checkpoint_interval', dest='checkpoint_interval',
                        help='number of iterations to display',
                        default=10000, type=int)

    parser.add_argument('--save_dir', dest='save_dir',
                        help='directory to save models', default="models", )
    parser.add_argument('--nw', dest='num_workers',
                        help='number of worker to load data',
                        default=0, type=int)
    parser.add_argument('--cuda', dest='cuda',
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

    # config optimization
    parser.add_argument('--o', dest='optimizer',
                        help='training optimizer',
                        default="sgd", type=str)
    parser.add_argument('--lr', dest='lr',
                        help='starting learning rate',
                        default=0.001, type=float)
    parser.add_argument('--lr_decay_step', dest='lr_decay_step',
                        help='step to do learning rate decay, unit is epoch',
                        default=5, type=int)
    parser.add_argument('--lr_decay_gamma', dest='lr_decay_gamma',
                        help='learning rate decay ratio',
                        default=0.1, type=float)

    # set training session
    parser.add_argument('--s', dest='session',
                        help='training session',
                        default=1, type=int)

    # resume trained model
    parser.add_argument('--r', dest='resume',
                        help='resume checkpoint or not',
                        default=False, type=bool)
    parser.add_argument('--checksession', dest='checksession',
                        help='checksession to load model',
                        default=1, type=int)
    parser.add_argument('--checkepoch', dest='checkepoch',
                        help='checkepoch to load model',
                        default=1, type=int)
    parser.add_argument('--checkpoint', dest='checkpoint',
                        help='checkpoint to load model',
                        default=0, type=int)
    # log and diaplay
    parser.add_argument('--use_tfboard', dest='use_tfboard',
                        help='whether use tensorflow tensorboard',
                        default=True, type=bool)
    parser.add_argument('--cascade', help='whether use cascade structure', default=1, action='store_true')

    args = parser.parse_args()
    return args


class sampler(Sampler):
    def __init__(self, train_size, batch_size):
        num_data = train_size
        self.num_per_batch = int(num_data / batch_size)
        self.batch_size = batch_size
        self.range = torch.arange(0, batch_size).view(1, batch_size).long()
        self.leftover_flag = False
        if num_data % batch_size:
            self.leftover = torch.arange(self.num_per_batch * batch_size, num_data).long()
            self.leftover_flag = True

    def __iter__(self):
        rand_num = torch.randperm(self.num_per_batch).view(-1, 1) * self.batch_size
        # rand_num = torch.arange(self.num_per_batch).long().view(-1, 1) * self.batch_size
        self.rand_num = rand_num.expand(self.num_per_batch, self.batch_size) + self.range

        self.rand_num_view = self.rand_num.view(-1)

        if self.leftover_flag:
            self.rand_num_view = torch.cat((self.rand_num_view, self.leftover), 0)

        return iter(self.rand_num_view)

    def __len__(self):
        return self.num_data


def _print(str, logger=None):
    print(str)
    if logger is None:
        return
    logger.info(str)


if __name__ == '__main__':

    args = parse_args()

    print('Called with args:')
    print(args)

      # if args.use_tfboard:
          # from model.utils.logger import Logger
          # # Set the logger
          # logger = Logger('./logs')
          # writer = SummaryWriter(comment=args.exp_name)


    if args.dataset == "pascal_voc":

      args.imdb_name = "voc_2007_train"
      args.imdbval_name = "voc_2007_test"
      args.set_cfgs = ['FPN_ANCHOR_SCALES', '[32, 64, 128, 256, 512]', 'FPN_FEAT_STRIDES', '[4, 8, 16, 32, 64]', 'MAX_NUM_GT_BOXES', '20']
    elif args.dataset == "pascal_voc_cap":
      args.imdb_name = "cap_voc_2007_train"
      args.imdbval_name = "cap_voc_2007_test"
      args.set_cfgs = ['FPN_ANCHOR_SCALES', '[32, 64, 128, 256, 512]', 'FPN_FEAT_STRIDES', '[4, 8, 16, 32, 64]', 'MAX_NUM_GT_BOXES', '20']
    elif args.dataset == "pascal_voc_bottle":
      args.imdb_name = "bottle_voc_2007_train"
      args.imdbval_name = "bottle_voc_2007_test"
      args.set_cfgs = ['FPN_ANCHOR_SCALES', '[32, 64, 128, 256, 512]', 'FPN_FEAT_STRIDES', '[4, 8, 16, 32, 64]', 'MAX_NUM_GT_BOXES', '20']
    elif args.dataset == "pascal_voc_0712":
      args.imdb_name = "voc_2007_trainval+voc_2012_trainval"
      args.imdbval_name = "voc_2007_test"
      args.set_cfgs = ['FPN_ANCHOR_SCALES', '[32, 64, 128, 256, 512]', 'FPN_FEAT_STRIDES', '[4, 8, 16, 32, 64]', 'MAX_NUM_GT_BOXES', '20']


    args.cfg_file = "cfgs/{}_ls.yml".format(args.net) if args.lscale else "cfgs/{}.yml".format(args.net)

    if args.cfg_file is not None:
      cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
      cfg_from_list(args.set_cfgs)

    args.cfg_file = "cfgs/{}_ls.yml".format(args.net) if args.lscale else "cfgs/{}.yml".format(args.net)

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    # print('trainval', cfg.POOLING_MODE)
    # print('fpn', get_cfg().POOLING_MODE)

    print('Using config:')
    pprint.pprint(cfg)
    # logging.info(cfg)
    np.random.seed(cfg.RNG_SEED)

    # torch.backends.cudnn.benchmark = True
    if torch.cuda.is_available() and not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    # train set
    # -- Note: Use validation set and disable the flipped to enable faster loading.
    cfg.TRAIN.USE_FLIPPED = True
    cfg.USE_GPU_NMS = args.cuda
    imdb, roidb, ratio_list, ratio_index = combined_roidb(args.imdb_name)
    train_size = len(roidb)

    # _print('{:d} roidb entries'.format(len(roidb)), logging)
    _print('{:d} roidb entries'.format(len(roidb)))

    # if args.exp_name is not None:
    #     output_dir = args.save_dir + "/" + args.net + "/" + args.dataset + '/' + args.exp_name
    # else:
    output_dir = args.save_dir + "/" + args.net + "/" + args.dataset
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    sampler_batch = sampler(train_size, args.batch_size)

    # for k, j in enumerate(ratio_index):
    #     if j == 23225:
    #         print(k)
    #         break

    dataset = roibatchLoader(roidb, ratio_list, ratio_index, args.batch_size, \
                             imdb.num_classes, training=True)
    # print('roidb', roidb[23225])
    # print(dataset[k][1])
    # print('--------')
    # print(dataset[k][2], dataset[k][3], dataset[k][4])
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size,
                                             sampler=sampler_batch, num_workers=args.num_workers)

    # initilize the tensor holder here.
    im_data = torch.FloatTensor(1)
    im_info = torch.FloatTensor(1)
    num_boxes = torch.LongTensor(1)
    gt_boxes = torch.FloatTensor(1)

    # ship to cuda
    if args.cuda:
        im_data = im_data.cuda()
        im_info = im_info.cuda()
        num_boxes = num_boxes.cuda()
        gt_boxes = gt_boxes.cuda()

    # make variable
    im_data = Variable(im_data)
    im_info = Variable(im_info)
    num_boxes = Variable(num_boxes)
    gt_boxes = Variable(gt_boxes)

    if args.cuda:
        cfg.CUDA = True

    # initilize the network here.
    if args.cascade:
        if args.net == 'res101':
            FPN = resnet(imdb.classes, 101, pretrained=True, class_agnostic=args.class_agnostic)
        else:
            print("network is not defined")
            pdb.set_trace()


    FPN.create_architecture()

    lr = cfg.TRAIN.LEARNING_RATE
    lr = args.lr
    # tr_momentum = cfg.TRAIN.MOMENTUM
    # tr_momentum = args.momentum

    params = []
    for key, value in dict(FPN.named_parameters()).items():
        if value.requires_grad:
            if 'bias' in key:
                params += [{'params': [value], 'lr': lr * (cfg.TRAIN.DOUBLE_BIAS + 1), \
                            'weight_decay': cfg.TRAIN.BIAS_DECAY and cfg.TRAIN.WEIGHT_DECAY or 0}]
            else:
                params += [{'params': [value], 'lr': lr, 'weight_decay': cfg.TRAIN.WEIGHT_DECAY}]

    if args.optimizer == "adam":
        lr = lr * 0.1
        optimizer = torch.optim.Adam(params)

    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(params, momentum=cfg.TRAIN.MOMENTUM)

    if args.resume:
        load_name = os.path.join(output_dir,
                                 'fpn_{}_{}_{}.pth'.format(args.checksession, args.checkepoch, args.checkpoint))
        _print("loading checkpoint %s" % (load_name), )
        checkpoint = torch.load(load_name)
        args.session = checkpoint['session']
        args.start_epoch = checkpoint['epoch']
        FPN.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr = optimizer.param_groups[0]['lr']
        if 'pooling_mode' in checkpoint.keys():
            cfg.POOLING_MODE = checkpoint['pooling_mode']
        _print("loaded checkpoint %s" % (load_name), )

    if args.mGPUs:
        FPN = nn.DataParallel(FPN)

    if args.cuda:
        FPN.cuda()

    iters_per_epoch = int(train_size / args.batch_size)

    for epoch in range(args.start_epoch, args.max_epochs):
        # setting to train mode
        FPN.train()
        loss_temp = 0
        start = time.time()

        if epoch % (args.lr_decay_step + 1) == 0:
            adjust_learning_rate(optimizer, args.lr_decay_gamma)
            lr *= args.lr_decay_gamma

        for step, data in enumerate(dataloader, 0):
            with torch.no_grad():
                  im_data.resize_(data[0].size()).copy_(data[0])
                  im_info.resize_(data[1].size()).copy_(data[1])
                  gt_boxes.resize_(data[2].size()).copy_(data[2])
                  num_boxes.resize_(data[3].size()).copy_(data[3])

            FPN.zero_grad()
            if args.cascade:
                _, _, _, rpn_loss_cls, rpn_loss_box, \
                RCNN_loss_cls, RCNN_loss_bbox, RCNN_loss_cls_2nd, RCNN_loss_bbox_2nd, RCNN_loss_cls_3rd, RCNN_loss_bbox_3rd, \
                roi_labels = FPN(im_data, im_info, gt_boxes, num_boxes)

                loss = rpn_loss_cls.mean() + rpn_loss_box.mean() \
                       + RCNN_loss_cls.mean() + RCNN_loss_bbox.mean() \
                       + RCNN_loss_cls_2nd.mean() + RCNN_loss_bbox_2nd.mean() \
                       + RCNN_loss_cls_3rd.mean() + RCNN_loss_bbox_3rd.mean()
            else:
                _, _, _, rpn_loss_cls, rpn_loss_box, RCNN_loss_cls, RCNN_loss_bbox, \
                roi_labels = FPN(im_data, im_info, gt_boxes, num_boxes)

                loss = rpn_loss_cls.mean() + rpn_loss_box.mean()+ RCNN_loss_cls.mean() + RCNN_loss_bbox.mean()

            loss_temp += loss.item()

            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % args.disp_interval == 0:
                end = time.time()
                if step > 0:
                    loss_temp /= args.disp_interval

                if args.mGPUs:
                    loss_rpn_cls = rpn_loss_cls.mean().item()
                    loss_rpn_box = rpn_loss_box.mean().item()
                    loss_rcnn_cls = RCNN_loss_cls.mean().item()
                    loss_rcnn_box = RCNN_loss_bbox.mean().item()

                    if args.cascade:
                        loss_rcnn_cls_2nd = RCNN_loss_cls_2nd.mean().item()
                        loss_rcnn_box_2nd = RCNN_loss_bbox_2nd.mean().item()
                        loss_rcnn_cls_3rd = RCNN_loss_cls_3rd.mean().item()
                        loss_rcnn_box_3rd = RCNN_loss_bbox_3rd.mean().item()

                    fg_cnt = torch.sum(roi_labels.data.ne(0))
                    bg_cnt = roi_labels.data.numel() - fg_cnt
                else:
                    loss_rpn_cls = rpn_loss_cls.item()
                    loss_rpn_box = rpn_loss_box.item()
                    loss_rcnn_cls = RCNN_loss_cls.item()
                    loss_rcnn_box = RCNN_loss_bbox.item()

                    if args.cascade:
                        loss_rcnn_cls_2nd = RCNN_loss_cls_2nd.item()
                        loss_rcnn_box_2nd = RCNN_loss_bbox_2nd.item()
                        loss_rcnn_cls_3rd = RCNN_loss_cls_3rd.item()
                        loss_rcnn_box_3rd = RCNN_loss_bbox_3rd.item()

                    fg_cnt = torch.sum(roi_labels.data.ne(0))
                    bg_cnt = roi_labels.data.numel() - fg_cnt

                _print("[session %d][epoch %2d][iter %4d/%4d] loss: %.4f, lr: %.2e" \
                       % (args.session, epoch, step, iters_per_epoch, loss_temp, lr), )
                _print("\t\t\tfg/bg=(%d/%d), time cost: %f" % (fg_cnt, bg_cnt, end - start), )

                if args.cascade:
                    _print("\t\t\trpn_cls: %.4f, rpn_box: %.4f, rcnn_cls: %.4f, rcnn_box %.4f, rcnn_cls_2nd: %.4f, "
                           "rcnn_box_2nd %.4f, rcnn_cls_3rd: %.4f, rcnn_box_3rd %.4f" % (loss_rpn_cls, loss_rpn_box,
                        loss_rcnn_cls, loss_rcnn_box, loss_rcnn_cls_2nd, loss_rcnn_box_2nd, loss_rcnn_cls_3rd, loss_rcnn_box_3rd), )
                else:
                    _print("\t\t\trpn_cls: %.4f, rpn_box: %.4f, rcnn_cls: %.4f, rcnn_box %.4f" \
                            % (loss_rpn_cls, loss_rpn_box, loss_rcnn_cls, loss_rcnn_box), )

                if args.use_tfboard:
                    if args.cascade:
                        scalars = [loss_temp, loss_rpn_cls, loss_rpn_box, loss_rcnn_cls, loss_rcnn_box, loss_rcnn_cls_2nd, loss_rcnn_box_2nd, loss_rcnn_cls_3rd, loss_rcnn_box_3rd]
                        names = ['loss', 'loss_rpn_cls', 'loss_rpn_box', 'loss_rcnn_cls', 'loss_rcnn_box', 'loss_rcnn_cls_2nd', 'loss_rcnn_box_2nd', 'loss_rcnn_cls_3rd', 'loss_rcnn_box_3rd']
                    else:
                        scalars = [loss_temp, loss_rpn_cls, loss_rpn_box, loss_rcnn_cls, loss_rcnn_box]
                        names = ['loss', 'loss_rpn_cls', 'loss_rpn_box', 'loss_rcnn_cls', 'loss_rcnn_box']
                    # write_scalars(writer, scalars, names, iters_per_epoch * (epoch - 1) + step, tag='train_loss')

                loss_temp = 0
                start = time.time()

        if args.mGPUs:
            save_name = os.path.join(output_dir, 'cascade_fpn_{}_{}_{}.pth'.format(args.session, epoch, step))
            save_checkpoint({
                'session': args.session,
                'epoch': epoch + 1,
                'model': FPN.module.state_dict(),
                'optimizer': optimizer.state_dict(),
                'pooling_mode': cfg.POOLING_MODE,
                'class_agnostic': args.class_agnostic,
            }, save_name)
        else:
            save_name = os.path.join(output_dir, 'cascade_fpn_{}_{}_{}.pth'.format(args.session, epoch, step))
            save_checkpoint({
                'session': args.session,
                'epoch': epoch + 1,
                'model': FPN.state_dict(),
                'optimizer': optimizer.state_dict(),
                'pooling_mode': cfg.POOLING_MODE,
                'class_agnostic': args.class_agnostic,
            }, save_name)
        _print('save model: {}'.format(save_name), )

        end = time.time()
        print(end - start)
