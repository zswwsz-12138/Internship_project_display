import os
import argparse
import numpy as np
from tqdm import tqdm
import pandas as pd
import joblib
from collections import OrderedDict

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder

from utils import *
import res2net


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default=None,
                        help='model name: (default: arch+timestamp)')
    parser.add_argument('--imagenet-dir', help='path to ImageNet directory')
    parser.add_argument('--arch', default='res2net50',
                        choices=res2net.__all__,
                        help='model architecture')
    parser.add_argument('--epochs', default=500, type=int)
    parser.add_argument('--lr', '--learning-rate', default=1e-1, type=float)
    parser.add_argument('--milestones', default='150,225', type=str)
    parser.add_argument('--gamma', default=0.1, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--weight-decay', default=5e-4, type=float)
    parser.add_argument('--nesterov', default=False, type=str2bool)

    args = parser.parse_args()

    return args


def train(args, train_loader, model, criterion, optimizer, epoch, scheduler=None):
    losses = AverageMeter()
    acc1s = AverageMeter()
    acc5s = AverageMeter()

    model.train()

    for i, (input, target) in tqdm(enumerate(train_loader), total=len(train_loader)):
        input = input.cuda()
        target = target.cuda()

        output = model(input)
        loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 1))

        losses.update(loss.item(), input.size(0))
        acc1s.update(acc1.item(), input.size(0))
        acc5s.update(acc5.item(), input.size(0))

        # compute gradient and do optimizing step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    log = OrderedDict([
        ('loss', losses.avg),
        ('acc1', acc1s.avg),
        ('acc5', acc5s.avg),
    ])

    return log


def validate(args, val_loader, model, criterion):
    losses = AverageMeter()
    acc1s = AverageMeter()
    acc5s = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        for i, (input, target) in tqdm(enumerate(val_loader), total=len(val_loader)):
            input = input.cuda()
            target = target.cuda()

            output = model(input)
            loss = criterion(output, target)

            acc1, acc5 = accuracy(output, target, topk=(1, 1))

            losses.update(loss.item(), input.size(0))
            acc1s.update(acc1.item(), input.size(0))
            acc5s.update(acc5.item(), input.size(0))

    log = OrderedDict([
        ('loss', losses.avg),
        ('acc1', acc1s.avg),
        ('acc5', acc5s.avg),
    ])

    return log


def main():
    args = parse_args()

    if args.name is None:
        args.name = '%s' %args.arch

    if not os.path.exists('models/%s' %args.name):
        os.makedirs('models/%s' %args.name)

    print('Config -----')
    for arg in vars(args):
        print('%s: %s' %(arg, getattr(args, arg)))
    print('------------')

    with open('models/%s/args.txt' %args.name, 'w') as f:
        for arg in vars(args):
            print('%s: %s' %(arg, getattr(args, arg)), file=f)

    joblib.dump(args, 'models/%s/args.pkl' %args.name)

    criterion = nn.CrossEntropyLoss().cuda()

    cudnn.benchmark = True

    # data loading code
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(64),  # 随机裁剪到224x224大小
        transforms.RandomHorizontalFlip(),  # 随机水平翻转
        transforms.RandomRotation(degrees=15),  # 随机旋转
        transforms.ToTensor(),  # 转化成Tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 正则化
    ])
    train_dataset = ImageFolder("./results/train/", transform=train_transform)  # 训练集数据
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=8, shuffle=True)  # 加载数据

    # 设置测试集
    test_transform = transforms.Compose([
        transforms.Resize((64, 64)),  # resize到224x224大小
        transforms.ToTensor(),  # 转化成Tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 正则化
    ])
    test_dataset = ImageFolder("./results/test/", transform=test_transform)  # 测试集数据
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=8, shuffle=False)  # 加载数据

    num_classes = 2

    # create model
    model = res2net.__dict__[args.arch]()
    model = model.cuda()

    optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr,
            momentum=args.momentum, weight_decay=args.weight_decay)

    scheduler = lr_scheduler.MultiStepLR(optimizer,
            milestones=[int(e) for e in args.milestones.split(',')], gamma=args.gamma)

    log = pd.DataFrame(index=[], columns=[
        'epoch', 'lr', 'loss', 'acc1', 'acc5', 'val_loss', 'val_acc1', 'val_acc5'
    ])

    best_acc = 0
    for epoch in range(args.epochs):
        print('Epoch [%d/%d]' %(epoch+1, args.epochs))

        scheduler.step()

        # train for one epoch
        train_log = train(args, train_loader, model, criterion, optimizer, epoch)
        # evaluate on validation set
        val_log = validate(args, test_loader, model, criterion)

        print('loss %.4f - acc1 %.4f - acc5 %.4f - val_loss %.4f - val_acc %.4f - val_acc5 %.4f'
            %(train_log['loss'], train_log['acc1'], train_log['acc5'], val_log['loss'], val_log['acc1'], val_log['acc5']))

        tmp = pd.Series([
            epoch,
            scheduler.get_lr()[0],
            train_log['loss'],
            train_log['acc1'],
            train_log['acc5'],
            val_log['loss'],
            val_log['acc1'],
            val_log['acc5'],
        ], index=['epoch', 'lr', 'loss', 'acc1', 'acc5', 'val_loss', 'val_acc1', 'val_acc5'])

        log = log.append(tmp, ignore_index=True)
        log.to_csv('models/%s/log.csv' %args.name, index=False)

        if val_log['acc1'] > best_acc and train_log['acc1'] >= 90:
            torch.save(model.state_dict(), 'models/%s/model.pth' %args.name)
            best_acc = val_log['acc1']
            print("=> saved best model")


if __name__ == '__main__':
    main()