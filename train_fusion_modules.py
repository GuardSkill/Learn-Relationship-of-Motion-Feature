from dataset.dataset import *
from torch.utils.data import Dataset, DataLoader
from torch import nn
from torch import optim
from torch.optim import lr_scheduler
import getpass
import os
import socket
import numpy as np
from dataset.preprocess_data import *
import torch
from models.model import generate_model
from models.resnext import Attention, fusion_model, new_fusion_model
from opts import parse_opts
from torch.autograd import Variable
import torch.nn.functional as F
import time
import sys
from utils import AverageMeter, calculate_accuracy, calculate_accuracy_video, Logger_MARS
import random
import pdb


def train():
    opt = parse_opts()
    print(opt)
    opt.arch = '{}-{}'.format(opt.model, opt.model_depth)
    torch.manual_seed(opt.manual_seed)

    print("Preprocessing train data ...")
    train_data = globals()['{}_test'.format(opt.dataset)](split=opt.split, train=1, opt=opt)
    print("Length of train data = ", len(train_data))

    print("Preprocessing validation data ...")
    val_data = globals()['{}_test'.format(opt.dataset)](split=opt.split, train=2, opt=opt)
    print("Length of validation data = ", len(val_data))

    if opt.modality=='RGB': opt.input_channels = 3
    elif opt.modality=='Flow': opt.input_channels = 2

    print("Preparing datatloaders ...")
    train_dataloader = DataLoader(train_data, batch_size = opt.batch_size, shuffle=True, num_workers = opt.n_workers, pin_memory = True, drop_last=True)
    val_dataloader   = DataLoader(val_data, batch_size = opt.batch_size, shuffle=True, num_workers = opt.n_workers, pin_memory = True, drop_last=True)
    print("Length of train datatloader = ",len(train_dataloader))
    print("Length of validation datatloader = ",len(val_dataloader))

    log_path = os.path.join(opt.result_path, opt.dataset)
    if not os.path.exists(log_path):
        os.makedirs(log_path)

    result_path = "{}/{}/".format(opt.result_path, opt.dataset)
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    if opt.log == 1:
        epoch_logger = Logger_MARS(os.path.join(log_path, 'Fusion_{}_{}_train_batch{}_sample{}_clip{}_lr{}_nesterov{}_manualseed{}_model{}{}_ftbeginidx{}_alpha{}.log'
            .format(opt.dataset, opt.split, opt.batch_size, opt.sample_size, opt.sample_duration, opt.learning_rate, opt.nesterov, opt.manual_seed, opt.model, opt.model_depth, opt.ft_begin_index
                    , opt.MARS_alpha))
                        ,['epoch', 'loss', 'acc', 'lr'], opt.MARS_resume_path, opt.begin_epoch)
        val_logger   = Logger_MARS(os.path.join(log_path, 'Fusion_{}_{}_val_batch{}_sample{}_clip{}_lr{}_nesterov{}_manualseed{}_model{}{}_ftbeginidx{}_alpha{}.log'
                        .format(opt.dataset,opt.split,  opt.batch_size, opt.sample_size, opt.sample_duration, opt.learning_rate, opt.nesterov, opt.manual_seed, opt.model, opt.model_depth, opt.ft_begin_index,
                             opt.MARS_alpha))
                        ,['epoch', 'loss', 'acc'], opt.MARS_resume_path, opt.begin_epoch)

    if opt.nesterov: dampening = 0
    else: dampening = opt.dampening

    # define the model 
    print("Loading models... ", opt.model, opt.model_depth)
    model1, parameters1 = generate_model(opt)

    # if testing RGB+Flow streams change input channels
    opt.input_channels = 2
    model2, parameters2 = generate_model(opt)
    model_fusion = new_fusion_model(opt.n_finetune_classes)
    model_fusion = model_fusion.cuda()
    model_fusion = nn.DataParallel(model_fusion)

    if opt.resume_path1:
        print('Loading MARS model {}'.format(opt.resume_path1))
        checkpoint = torch.load(opt.resume_path1)
        assert opt.arch == checkpoint['arch']
        model1.load_state_dict(checkpoint['state_dict'])
    if opt.resume_path2:
        print('Loading Flow model {}'.format(opt.resume_path2))
        checkpoint = torch.load(opt.resume_path2)
        assert opt.arch == checkpoint['arch']
        model2.load_state_dict(checkpoint['state_dict'])

    if opt.resume_path3:
        print('Loading Fusion model {}'.format(opt.resume_path3))
        checkpoint = torch.load(opt.resume_path3)
        assert opt.arch == checkpoint['arch']
        model2.load_state_dict(checkpoint['state_dict'])

    model1.eval()
    model2.eval()
    model_fusion.train()
    for p in model1.parameters():
        # if p.requires_grad:
        #     print("Need to freeze the parameters")
        p.requires_grad = False
    for p in model2.parameters():
        # if p.requires_grad:
        #     print("Need to freeze the parameters..")
        p.requires_grad = False

    print("Initializing the optimizer ...")

    print("lr = {} \t momentum = {} \t dampening = {} \t weight_decay = {}, \t nesterov = {}"
          .format(opt.learning_rate, opt.momentum, dampening, opt.weight_decay, opt.nesterov))
    print("LR patience = ", opt.lr_patience)

    optimizer = optim.SGD(
        model_fusion.parameters(),
        lr=opt.learning_rate,
        momentum=opt.momentum,
        dampening=dampening,
        weight_decay=opt.weight_decay,
        nesterov=opt.nesterov)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=opt.lr_patience)

    criterion = nn.CrossEntropyLoss().cuda()
    print('run')
    for epoch in range(opt.begin_epoch, opt.n_epochs + 1):
        batch_time = AverageMeter()
        data_time  = AverageMeter()
        losses = AverageMeter()
        accuracies = AverageMeter()
        weights=AverageMeter()
        end_time = time.time()
        for i, (inputs, targets) in enumerate(train_dataloader):
            data_time.update(time.time() - end_time)
            inputs_MARS = inputs[:, 0:3, :, :, :]
            inputs_Flow = inputs[:, 3:, :, :, :]

            targets = targets.cuda(non_blocking=True)
            inputs_MARS = Variable(inputs_MARS)
            inputs_Flow = Variable(inputs_Flow)
            targets = Variable(targets)
            outputs_MARS = model1(inputs_MARS)
            outputs_Flow = model2(inputs_Flow)

            weight,outputs_var =model_fusion(outputs_MARS.detach(),outputs_Flow.detach())
            loss=criterion(outputs_var,targets)
            acc = calculate_accuracy(outputs_var, targets)

            losses.update(loss.data, inputs.size(0))
            accuracies.update(acc, inputs.size(0))
            weights.update(weight[0][0].data, inputs.size(0))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            batch_time.update(time.time() - end_time)
            end_time = time.time()

            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc {acc.val:.3f} ({acc.avg:.3f})\t'
                  'Weight {weight.val:.3f} ({weight.avg:.3f})'.format(
                epoch,
                i + 1,
                len(train_dataloader),
                batch_time=batch_time,
                data_time=data_time,
                loss=losses,
                acc=accuracies,
                weight=weights
            ))

        if opt.log == 1:
            epoch_logger.log({
                'epoch': epoch,
                'loss': losses.avg,
                'acc': accuracies.avg,
                'lr': optimizer.param_groups[0]['lr']
            })

        if epoch % opt.checkpoint == 0:
            if opt.pretrain_path != '':
                save_file_path = os.path.join(log_path,
                                              'Fusion_{}_{}_train_batch{}_sample{}_clip{}_lr{}_nesterov{}_manualseed{}_model{}{}_ftbeginidx{}_alpha{}_{}.pth'
                                              .format(opt.dataset, opt.split, opt.batch_size, opt.sample_size,
                                                      opt.sample_duration, opt.learning_rate, opt.nesterov,
                                                      opt.manual_seed, opt.model, opt.model_depth,
                                                      opt.ft_begin_index,
                                                      opt.MARS_alpha, epoch))
            else:
                save_file_path = os.path.join(log_path,
                                              'Fusion_{}_{}_train_batch{}_sample{}_clip{}_lr{}_nesterov{}_manualseed{}_model{}{}_ftbeginidx{}_alpha{}_{}.pth'
                                              .format(opt.dataset, opt.split, opt.batch_size, opt.sample_size,
                                                      opt.sample_duration, opt.learning_rate, opt.nesterov,
                                                      opt.manual_seed, opt.model, opt.model_depth,
                                                      opt.ft_begin_index,
                                                      opt.MARS_alpha, epoch))
            states = {
                'epoch': epoch + 1,
                'arch': opt.arch,
                'state_dict': model_fusion.state_dict(),
                'optimizer': optimizer.state_dict(),
            }
            torch.save(states, save_file_path)

        model_fusion.eval()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        accuracies = AverageMeter()

        end_time = time.time()
        with torch.no_grad():
            for i, (inputs, targets) in enumerate(val_dataloader):

                data_time.update(time.time() - end_time)
                inputs_MARS = inputs[:, 0:3, :, :, :]
                inputs_Flow = inputs[:, 3:, :, :, :]

                targets = targets.cuda(non_blocking=True)
                inputs_MARS = Variable(inputs_MARS)
                inputs_Flow=Variable(inputs_Flow)
                targets = Variable(targets)

                outputs_MARS = model1(inputs_MARS)
                outputs_Flow = model2(inputs_Flow)
                _,outputs_var=model_fusion(outputs_MARS,outputs_Flow)
                loss = criterion(outputs_var, targets)
                acc = calculate_accuracy(outputs_var, targets)

                losses.update(loss.data, inputs.size(0))
                accuracies.update(acc, inputs.size(0))

                batch_time.update(time.time() - end_time)
                end_time = time.time()

                print('Val_Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc {acc.val:.3f} ({acc.avg:.3f})'.format(
                    epoch,
                    i + 1,
                    len(val_dataloader),
                    batch_time=batch_time,
                    data_time=data_time,
                    loss=losses,
                    acc=accuracies))

        if opt.log == 1:
            val_logger.log({'epoch': epoch, 'loss': losses.avg, 'acc': accuracies.avg})
        scheduler.step(losses.avg)
    
if __name__=="__main__":
    train()
