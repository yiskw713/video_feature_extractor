import adabound
import argparse
import os
import pandas as pd
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
import yaml

from addict import Dict
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, RandomCrop, Normalize

from utils.checkpoint import save_checkpoint, resume
from utils.class_weight import get_class_weight
from utils.dataset import Kinetics
from utils.mean import get_mean, get_std
from model import resnet
from model import slowfast
from model.metric import L2ConstrainedLinear
from model.msc import SpatialMSC, TemporalMSC, SpatioTemporalMSC
from model import resnext


def get_arguments():
    '''
    parse all the arguments from command line inteface
    return a list of parsed arguments
    '''

    parser = argparse.ArgumentParser(
        description='train a network for action recognition')
    parser.add_argument('config', type=str, help='path of a config file')
    parser.add_argument('--resume', action='store_true',
                        help='Add --resume option if you start training from checkpoint.')

    return parser.parse_args()


""" codes for training """


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
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

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions"""
    N = output.shape[0]
    maxk = max(topk)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k)
    return N, res


def train(train_loader, model, criterion, optimizer, epoch, config, device):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        images = images.to(device)
        target = target.to(device)

        # compute output
        output = model(images)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # show progress bar per 1000 iteration
        if i % 1000 == 0:
            progress.display(i)

    return losses.avg, top1.avg, top5.avg


def validate(val_loader, model, criterion, config, device):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            images = images.to(device)
            target = target.to(device)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # show progress bar per 1000 iteration
            if i % 1000 == 0:
                progress.display(i)

    # TODO: 各GPUの出力をどう揃えるか
    return losses.avg, top1.avg, top5.avg


def main():
    args = get_arguments()

    # configuration
    CONFIG = Dict(yaml.safe_load(open(args.config)))

    # writer
    if CONFIG.writer_flag:
        writer = SummaryWriter(CONFIG.result_path)
    else:
        writer = None

    # DataLoaders
    normalize = Normalize(mean=get_mean(), std=get_std())

    train_data = Kinetics(
        CONFIG,
        transform=Compose([
            RandomCrop((CONFIG.height, CONFIG.width)),
            ToTensor(),
            normalize,
        ])
    )

    val_data = Kinetics(
        CONFIG,
        transform=Compose([
            RandomCrop((CONFIG.height, CONFIG.width)),
            ToTensor(),
            normalize,
        ]),
        mode='validation'
    )

    train_loader = DataLoader(
        train_data,
        batch_size=CONFIG.batch_size,
        shuffle=True,
        num_workers=CONFIG.num_workers,
        drop_last=True
    )

    val_loader = DataLoader(
        val_data,
        batch_size=CONFIG.batch_size,
        shuffle=False,
        num_workers=CONFIG.num_workers
    )

    # load model
    print('\n------------------------Loading Model------------------------\n')

    if CONFIG.model == 'resnet18':
        print(CONFIG.model + ' will be used as a model.')
        model = resnet.generate_model(18, n_classes=CONFIG.n_classes)
    elif CONFIG.model == 'resnet50':
        print('ResNext101 will be used as a model.')
        model = resnext.generate_model(101, n_classes=CONFIG.n_classes)
    else:
        print('resnet18 will be used as a model.')
        model = resnet.generate_model(18, n_classes=CONFIG.n_classes)

    # metric
    if CONFIG.metric == 'L2constrain':
        print('L2constrain metric will be used.')
        model.fc = L2ConstrainedLinear(
            model.fc.in_features, model.fc.out_features)

    # multi-scale input
    if CONFIG.msc == 'Temporal':
        print('Temporal multi-scale input will be used')
        model = TemporalMSC(model)
    elif CONFIG.msc == 'Spatial':
        print('Spatial multi-scale input will be used')
        model = SpatialMSC(model)
    elif CONFIG.msc == 'SpatioTemporal':
        print('SpatioTemporal multi-scale input will be used')
        model = SpatioTemporalMSC(model)

    # set optimizer, lr_scheduler
    if CONFIG.optimizer == 'Adam':
        print(CONFIG.optimizer + ' will be used as an optimizer.')
        optimizer = optim.Adam(model.parameters(), lr=CONFIG.learning_rate)
    elif CONFIG.optimizer == 'SGD':
        print(CONFIG.optimizer + ' will be used as an optimizer.')
        optimizer = optim.SGD(
            model.parameters(),
            lr=CONFIG.learning_rate,
            momentum=CONFIG.momentum,
            dampening=CONFIG.dampening,
            weight_decay=CONFIG.weight_decay,
            nesterov=CONFIG.nesterov)
    elif CONFIG.optimizer == 'AdaBound':
        print(CONFIG.optimizer + ' will be used as an optimizer.')
        optimizer = adabound.AdaBound(
            model.parameters(), lr=CONFIG.learning_rate, final_lr=CONFIG.final_lr, weight_decay=CONFIG.weight_decay)
    else:
        print('There is no optimizer which suits to your option. \
            Instead, SGD will be used as an optimizer.')
        optimizer = optim.SGD(
            model.parameters(),
            lr=CONFIG.learning_rate,
            momentum=CONFIG.momentum,
            dampening=CONFIG.dampening,
            weight_decay=CONFIG.weight_decay,
            nesterov=CONFIG.nesterov)

    # learning rate scheduler
    if CONFIG.optimizer == 'SGD':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 'min', patience=CONFIG.lr_patience)
    else:
        scheduler = None

    # send the model to cuda/cpu
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    if device == 'cuda':
        model = torch.nn.DataParallel(model)  # make parallel
        torch.backends.cudnn.benchmark = True
    else:
        print('You have to use GPUs because training 3DCNN is computationally expensive.')
        sys.exit(1)

    # resume if you want
    begin_epoch = 0
    log = None
    if args.resume:
        if os.path.exists(os.path.join(CONFIG.result_path, 'checkpoint.pth')):
            print('loading the checkpoint...')
            begin_epoch, model, optimizer, scheduler = resume(
                CONFIG, model, optimizer, scheduler)
            print('training will start from {} epoch'.format(begin_epoch))
        if os.path.exists(os.path.join(CONFIG.result_path, 'log.csv')):
            log = pd.read_csv(os.path.join(CONFIG.result_path, 'log.csv'))

    # generate log when you start training from scratch
    if log is None:
        log = pd.DataFrame(
            columns=['epoch', 'lr', 'train_loss', 'val_loss', 'acc@1', 'acc@5']
        )

    # criterion for loss
    if CONFIG.class_weight:
        criterion = nn.CrossEntropyLoss(
            weight=get_class_weight().to(device))
    else:
        criterion = nn.CrossEntropyLoss()

    # train and validate model
    print('\n------------------------Start training------------------------\n')
    losses_train = []
    losses_val = []
    top1_accuracy = []
    top5_accuracy = []
    best_top1_accuracy = 0.0
    best_top5_accuracy = 0.0

    for epoch in range(begin_epoch, CONFIG.max_epoch):

        # training
        loss_train = train(
            model, train_loader, criterion, optimizer, CONFIG, device)
        losses_train.append(loss_train)

        # validation
        loss_val, top1, top5 = validation(
            model, val_loader, criterion, CONFIG, device)

        if CONFIG.optimizer == 'SGD':
            scheduler.step(loss_val)

        losses_val.append(loss_val)
        top1_accuracy.append(top1)
        top5_accuracy.append(top5)

        # save a model if topk accuracy is higher than ever
        # save base models, NOT DataParalled models
        if best_top1_accuracy < top1_accuracy[-1]:
            best_top1_accuracy = top1_accuracy[-1]
            torch.save(
                model.module.state_dict(), os.path.join(CONFIG.result_path, 'best_top1_accuracy_model.prm'))

        if best_top5_accuracy < top5_accuracy[-1]:
            best_top5_accuracy = top5_accuracy[-1]
            torch.save(
                model.module.state_dict(), os.path.join(CONFIG.result_path, 'best_top5_accuracy_model.prm'))

        # save checkpoint every epoch
        save_checkpoint(CONFIG, epoch, model, optimizer, scheduler)

        # save a model every 10 epoch
        # save base models, NOT DataParalled models
        if epoch % 10 == 0 and epoch != 0:
            torch.save(
                model.module.state_dict(), os.path.join(CONFIG.result_path, 'epoch_{}_model.prm'.format(epoch)))

        # tensorboardx
        if writer is not None:
            writer.add_scalar("loss_train", losses_train[-1], epoch)
            writer.add_scalar('loss_val', losses_val[-1], epoch)
            writer.add_scalars("iou", {
                'top1_accuracy': top1_accuracy[-1],
                'top5_accuracy': top5_accuracy[-1]}, epoch)

        # write logs to dataframe and csv file
        tmp = pd.Series([
            epoch,
            scheduler.get_lr()[0],
            losses_train[-1],
            losses_val[-1],
            top1_accuracy[-1],
            top5_accuracy[-1],
        ], index=log.columns)

        log = log.append(tmp, ignore_index=True)
        log.to_csv(os.path.join(CONFIG.result_path, 'log.csv'), index=False)

        print(
            'epoch: {}\tloss train: {:.5f}\tloss val: {:.5f}\ttop1_accuracy: {:.5f}\ttop5_accuracy: {:.5f}'
            .format(epoch, losses_train[-1], losses_val[-1], top1_accuracy[-1], top5_accuracy[-1])
        )

    # save base models, NOT DataParalled models
    torch.save(
        model.module.state_dict(), os.path.join(CONFIG.result_path, 'final_model.prm'))


if __name__ == '__main__':
    main()
