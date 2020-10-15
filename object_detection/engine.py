'''This script is a slighty modified version of the engine script provided in the
Torchvision Object Detection Finetuning Tutorial: https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html
Command to view tensorboard: tensorboard --logdir=runs then navigate to https://localhost:6006 
'''
import math
import sys
import time
import torch
import torchvision.models.detection.mask_rcnn

import dataset_interface.object_detection.utils as utils
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter()

def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq, loss_file_name=None):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1. / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = utils.warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)

    for images, targets in metric_logger.log_every(data_loader, print_freq, header, loss_file_name):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)

        losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    writer.add_scalar('Train_Loss/Classifier_loss',loss_dict_reduced['loss_classifier'].item(),epoch)
    writer.add_scalar('Train_Loss/Box_reg_loss',loss_dict_reduced['loss_box_reg'].item(),epoch)
    writer.add_scalar('Train_Loss/Objectness_loss',loss_dict_reduced['loss_objectness'].item(),epoch)
    writer.add_scalar('Train_Loss/rpn_box_reg_loss',loss_dict_reduced['loss_rpn_box_reg'].item(),epoch)

def validate_one_epoch(model, data_loader, device, epoch, print_freq, loss_file_name=None):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")

    header = '------------------------------\n'
    header += 'Epoch: [{}]'.format(epoch)

    for images, targets in metric_logger.log_every(data_loader, print_freq, header, loss_file_name):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)

        losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
    writer.add_scalar('Val_Loss/Classifier_loss',loss_dict_reduced['loss_classifier'].item(),epoch)
    writer.add_scalar('Val_Loss/Box_reg_loss',loss_dict_reduced['loss_box_reg'].item(),epoch)
    writer.add_scalar('Val_Loss/Objectness_loss',loss_dict_reduced['loss_objectness'].item(),epoch)
    writer.add_scalar('Val_Loss/rpn_box_reg_loss',loss_dict_reduced['loss_rpn_box_reg'].item(),epoch)
