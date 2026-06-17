# -*- coding: utf-8 -*-
"""
Created on Thu Mar 30 11:39:11 2023

@author: nguyen
"""
import os
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"  # specify which GPU(s) to be used
#os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"  # specify which GPU(s) to be used
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # specify which GPU(s) to be used

import argparse

import random
import shutil
import time
import warnings

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from models.SSFNet import *
import writeLogAcc as wA
from models.cross_entropy import LabelSmoothingCrossEntropy
import torchvision

cudnn.benchmark = True 

from ptflops import get_model_complexity_info

path_ImageNet = '../../datasets/places365'

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('-r', '--data', type=str, default='', help='path to dataset')
parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=5, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--world-size', default=1, type=int,
                    help='number of distributed processes')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='gloo', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int, nargs='+',
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--ksize', default=None, type=list,
                    help='Manually select the eca module kernel size')
parser.add_argument('--action', default='', type=str,
                    help='other information.')
parser.add_argument('--download', action='store_true', help='Download the specified dataset before running the training.')
                    

#best_prec1 = 0


def main():
    global args, best_prec1
    args = parser.parse_args()    
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')
    #args.gpu = 1
    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    #args.distributed = args.world_size > 1
    args.distributed = False

    if args.distributed:
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size)
    torch.autograd.set_detect_anomaly(True)
    # create model
    args.data = path_ImageNet

    
    args.arch = 'Places365' + '_SSFNet_final'  
    #pathout = './checkpoints/' + strmode
    directory = "checkpoints/%s/"%(args.arch + '_' + args.action)
    if not os.path.exists(directory):
        os.makedirs(directory)
    filenameLOG = "./checkpoints/%s/"%(args.arch + '_' + args.action) + '/' + args.arch + '.txt'
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
        model = models.__dict__[args.arch](k_size=args.ksize, pretrained=True)
    else:            
        model = build_SSFNet(num_classes=365, width_multiplier=1.0, cifar=False, groups=1)
            
    if args.gpu is not None:
        model = model.cuda(args.gpu)
    elif args.distributed:
        model.cuda()
        model = torch.nn.parallel.DistributedDataParallel(model)
    else:
        model = torch.nn.DataParallel(model).cuda()

    print(model)
    
    # get the number of models parameters
    print('Number of models parameters: {}'.format(
        sum([p.data.nelement() for p in model.parameters()])))

    file_log_model = directory + '_modelLog.txt'
    with open(file_log_model, 'w') as f:
        f.write(str(model))
        f.write('\n')
        f.write('Number of model parameters: {}'.format(
        sum([p.data.nelement() for p in model.parameters()])))
        f.write('\n')
    
        macs, params = get_model_complexity_info(model, (3, 224, 224), as_strings=True,
                                        print_per_layer_stat=True, verbose=True, flops_units='GMac',
                                        param_units='M')
        # print_per_layer_stat=True, verbose=True,param_units='M')

        # , flops_units='GMac')
        macs1 = macs.split()
        strmacs1 = str(float(macs1[0]) / 2) + ' ' + macs1[1][0]
        
        f.write('{:<30}  {:<8}'.format('Floating-point operations (FLOPs): ', strmacs1))
        f.write('\n')
        f.write('{:<30}  {:<8}'.format('Number of parameters using ptflops: ', params))
        f.write('\n')

    # define loss function (criterion) and optimizer
    train_loss_fn = LabelSmoothingCrossEntropy(smoothing=0.1).cuda(args.gpu)
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)                   
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if args.evaluate:
        pathcheckpoint = "./checkpoints/%s/"%(args.arch + '_' + args.action) + "model_best.pth.tar"
        if os.path.isfile(pathcheckpoint):
            print("=> loading checkpoint '{}'".format(pathcheckpoint))
            checkpoint = torch.load(pathcheckpoint)
            model.load_state_dict(checkpoint['state_dict'])
            #optimizer.load_state_dict(checkpoint['optimizer'])
            del checkpoint
        else:
            print("=> no checkpoint found at '{}'".format(pathcheckpoint))
            return
    if args.resume:
        path_resume = "./checkpoints/%s/"%(args.arch + '_' + args.action) + "checkpoint.pth.tar"
        if os.path.isfile(path_resume):
            print("--------------------------------")
            print("=> loading checkpoint '{}'".format(path_resume))
            checkpoint = torch.load(path_resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                    .format(args.resume, checkpoint['epoch']))
            del checkpoint
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    else:
        best_prec1 = 0
    cudnn.benchmark = True

    # Data loading code
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


    transforms_train =  transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    transforms_val = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])

    dataset_class_train = torchvision.datasets.Places365
    dataset_class_val = torchvision.datasets.Places365
    if args.download:
        dataset_class_train(root=path_ImageNet, split="train-standard",small = True, transform=transforms_train, download=True)
        dataset_class_val(root=path_ImageNet, split="val", transform=transforms_val, download=True)
    dataset_train = dataset_class_train(root=path_ImageNet,small = True, split="train-standard", transform=transforms_train)
    dataset_val = dataset_class_val(root=path_ImageNet, split="val", transform=transforms_val)

        
    train_loader = torch.utils.data.DataLoader( dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(dataset_val, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)      
    
    if args.evaluate:
        m = time.time()
        _, _ =validate(val_loader, model, criterion)
        n = time.time()
        print((n-m)/3600)
        return

    Loss_plot = {}
    train_prec1_plot = {}
    train_prec5_plot = {}
    val_prec1_plot = {}
    val_prec5_plot = {}
    epoch_max = None
    #best_prec1 = 0
    for epoch in range(args.start_epoch, args.epochs):
        start_time = time.time()
        adjust_learning_rate(optimizer, epoch)

        # train for one epoch
        loss_temp, train_prec1_temp, train_prec5_temp = train(train_loader, model, train_loss_fn, optimizer, epoch)
        
        Loss_plot[epoch] = loss_temp
        train_prec1_plot[epoch] = train_prec1_temp
        train_prec5_plot[epoch] = train_prec5_temp

        # evaluate on validation set
        prec1, prec5 = validate(val_loader, model, criterion)
        val_prec1_plot[epoch] = prec1
        val_prec5_plot[epoch] = prec5

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer' : optimizer.state_dict(),
        }, is_best)
        
        data_save(directory + 'Loss_plot.txt', Loss_plot)
        data_save(directory + 'train_prec1.txt', train_prec1_plot)
        data_save(directory + 'train_prec5.txt', train_prec5_plot)
        data_save(directory + 'val_prec1.txt', val_prec1_plot)
        data_save(directory + 'val_prec5.txt', val_prec5_plot)
        
        line = 'Epoch {}/{} summary: loss_train={:.5f}, acc_train={:.2f}%, loss_val={:.2f}, acc_val={:.2f}% (best: {:.2f}% @ epoch {})'.format(epoch, args.epochs, loss_temp, train_prec1_temp, 0, prec1, best_prec1, epoch_max)
        wA.writeLogAcc(filenameLOG,line)
        end_time = time.time()
        time_value = (end_time - start_time) / 3600
        print("-" * 80)
        print(time_value)
        print("-" * 80)


def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    losses_batch = {}
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            input = input.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)

        # compute output
        output = model(input)
        loss = criterion(output, target)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
     
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, top5=top5))

    return losses.avg, top1.avg, top5.avg


def validate(val_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            if args.gpu is not None:
                input = input.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(prec1[0], input.size(0))
            top5.update(prec5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                       i, len(val_loader), batch_time=batch_time, loss=losses,
                       top1=top1, top5=top5))

        print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top1.avg, top5.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    directory = "checkpoints/%s/"%(args.arch + '_' + args.action)
    
    filename = directory + filename
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, directory + 'model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:            
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def data_save(root, file):
    if not os.path.exists(root):
        os.mknod(root)
    file_temp = open(root, 'r')
    lines = file_temp.readlines()
    if not lines:
        epoch = -1
    else:
        epoch = lines[-1][:lines[-1].index(' ')]
    epoch = int(epoch)
    file_temp.close()
    file_temp = open(root, 'a')
    for line in file:
        if line > epoch:
            file_temp.write(str(line) + " " + str(file[line]) + '\n')
    file_temp.close()


if __name__ == '__main__':
    main()
