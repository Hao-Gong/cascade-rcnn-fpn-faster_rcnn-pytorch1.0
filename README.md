# Pytorch 1.0 Implementation of Faster R-CNN + FPN + Cascade Faster R-CNN
This repo supports Faster R-CNN, FPN and Cascade Faster R-CNN based on pyTorch 1.0. Additionally deformable convolutional layer is also support! 

## Train 

## Train Faster R-CNN on VOC dataset and pretrained resnet101.pth model
Before training, set the right directory to save and load the trained models. Change the arguments "save_dir" and "load_dir" in trainval_net.py and test_net.py to adapt to your environment.

```
python python train_faster_rcnn.py --dataset pascal_voc--net res101 --bs 1 --nw 1 --lr 0.01 --lr_decay_step 8 --cuda
```

Change dataset to "coco" or 'vg' if you want to train on COCO or Visual Genome.

## Train FPN on VOC dataset and pretrained resnet101.pth model
Before training, set the right directory to save and load the trained models. Change the arguments "save_dir" and "load_dir" in trainval_net.py and test_net.py to adapt to your environment.

The learning rate should be lower than Faster RCNN

```
python python train_fpn.py --dataset pascal_voc--net res101 --bs 1 --nw 1 --lr 0.001 --lr_decay_step 8 --cuda
```

Change dataset to "coco" or 'vg' if you want to train on COCO or Visual Genome.

## Train Cascade RCNN on VOC dataset and pretrained resnet101.pth model
Before training, set the right directory to save and load the trained models. Change the arguments "save_dir" and "load_dir" in trainval_net.py and test_net.py to adapt to your environment.

The learning rate should be lower than Faster RCNN

```
python python train_cascade_fpn.py --dataset pascal_voc--net res101 --bs 1 --nw 1 --lr 0.001 --lr_decay_step 8 --cuda
```

Change dataset to "coco" or 'vg' if you want to train on COCO or Visual Genome.

## Test Faster R-CNN and generate json outputs

If you want to evlauate the detection performance of a pre-trained res101 model on pascal_voc test set, simply run
```
python json_test_faster_rcnn.py --dataset pascal_voc --net res101 \
                   --checksession $SESSION --checkepoch $EPOCH --checkpoint $CHECKPOINT \
                   --cuda
```
Specify the specific model session, chechepoch and checkpoint, e.g., SESSION=1, EPOCH=6, CHECKPOINT=416.

## Test FPN and generate json outputs

If you want to evlauate the detection performance of a pre-trained res101 model on pascal_voc test set, simply run
```
python json_test_fpn.py --dataset pascal_voc --net res101 \
                   --checksession $SESSION --checkepoch $EPOCH --checkpoint $CHECKPOINT \
                   --cuda
```
Specify the specific model session, chechepoch and checkpoint, e.g., SESSION=1, EPOCH=6, CHECKPOINT=416.


## Test Cascade R-CNN and generate json outputs

If you want to evlauate the detection performance of a pre-trained res101 model on pascal_voc test set, simply run
```
python json_test_cascade_fpn.py --dataset pascal_voc --net res101 \
                   --checksession $SESSION --checkepoch $EPOCH --checkpoint $CHECKPOINT \
                   --cuda
```
Specify the specific model session, chechepoch and checkpoint, e.g., SESSION=1, EPOCH=6, CHECKPOINT=416.

## Citation

    @article{jjfaster2rcnn,
        Author = {Jianwei Yang and Jiasen Lu and Dhruv Batra and Devi Parikh},
        Title = {A Faster Pytorch Implementation of Faster R-CNN},
        Journal = {https://github.com/jwyang/faster-rcnn.pytorch},
        Year = {2017}
    }

    @inproceedings{renNIPS15fasterrcnn,
        Author = {Shaoqing Ren and Kaiming He and Ross Girshick and Jian Sun},
        Title = {Faster {R-CNN}: Towards Real-Time Object Detection
                 with Region Proposal Networks},
        Booktitle = {Advances in Neural Information Processing Systems ({NIPS})},
        Year = {2015}
    }
