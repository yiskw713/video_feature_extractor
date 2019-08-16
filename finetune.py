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
from models import resnet
from models import i3d


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
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


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
    for i, sample in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        x = sample['clip']
        t = sample['cls_id']

        x = x.to(device)
        t = t.to(device)

        # compute output
        output = model(x)
        loss = criterion(output, t)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, t, topk=(1, 5))
        losses.update(loss.item(), x.size(0))
        top1.update(acc1[0].item(), x.size(0))
        top5.update(acc5[0].item(), x.size(0))

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

    return losses, top1.avg, top5.avg


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
        for i, sample in enumerate(val_loader):
            x = sample['clip']
            t = sample['cls_id']
            x = x.to(device)
            t = t.to(device)

            # compute output
            output = model(x)
            loss = criterion(output, t)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, t, topk=(1, 5))
            losses.update(loss.item(), x.size(0))
            top1.update(acc1[0].item(), x.size(0))
            top5.update(acc5[0].item(), x.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # show progress bar per 1000 iteration
            if i % 1000 == 0:
                progress.display(i)

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
        ]),
        mode='training'
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
        print('ResNet18 will be used as a model.')
        model = resnet.generate_model(18, n_classes=CONFIG.n_classes)
    elif CONFIG.model == 'resnet50':
        print('ResNet50 will be used as a model.')
        model = resnet.generate_model(50, n_classes=CONFIG.n_classes)
    elif CONFIG.model == 'i3d':
        print('I3D will be used as a model.')
        model = i3d.InceptionI3d(num_classes=CONFIG.n_classes, in_channels=3)
    else:
        print('There is no model appropriate to your choice. '
              'Instead, resnet18 will be used as a model.')
        model = resnet.generate_model(18, n_classes=CONFIG.n_classes)

    # send the model to cuda/cpu
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    if device == 'cuda':
        torch.backends.cudnn.benchmark = True
    else:
        print('You have to use GPUs because training 3DCNN is computationally expensive.')
        sys.exit(1)

    # load pretrained model
    if CONFIG.pretrained_weights is not None:
        if CONFIG.model == 'i3d':
            state_dict = torch.load(CONFIG.pretrained_weights)
            model.load_state_dict(state_dict, strict=False)
        else:
            state_dict = torch.load(CONFIG.pretrained_weights)
            model.load_state_dict(state_dict, strict=False)

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
            nesterov=CONFIG.nesterov
        )
    elif CONFIG.optimizer == 'AdaBound':
        print(CONFIG.optimizer + ' will be used as an optimizer.')
        optimizer = adabound.AdaBound(
            model.parameters(),
            lr=CONFIG.learning_rate,
            final_lr=CONFIG.final_lr,
            weight_decay=CONFIG.weight_decay
        )
    else:
        print('There is no optimizer which suits to your option. \
            Instead, SGD will be used as an optimizer.')
        optimizer = optim.SGD(
            model.parameters(),
            lr=CONFIG.learning_rate,
            momentum=CONFIG.momentum,
            dampening=CONFIG.dampening,
            weight_decay=CONFIG.weight_decay,
            nesterov=CONFIG.nesterov
        )

    # learning rate scheduler
    if CONFIG.optimizer == 'SGD':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 'min', patience=CONFIG.lr_patience
        )
    else:
        scheduler = None

    # resume if you want
    begin_epoch = 0
    best_acc1 = 0
    log = None
    if args.resume:
        if os.path.exists(os.path.join(CONFIG.result_path, 'checkpoint.pth')):
            print('loading the checkpoint...')
            begin_epoch, model, optimizer, best_acc1, scheduler = resume(
                CONFIG, model, optimizer, scheduler)
            print('training will start from {} epoch'.format(begin_epoch))
        else:
            print("there is no checkpoint at the result folder")
        if os.path.exists(os.path.join(CONFIG.result_path, 'log.csv')):
            print('loading the log file...')
            log = pd.read_csv(os.path.join(CONFIG.result_path, 'log.csv'))
        else:
            print("there is no log file at the result folder.")
            print('Making a log file...')
            log = pd.DataFrame(
                columns=['epoch', 'lr', 'train_loss', 'val_loss', 'train_acc@1',
                         'train_acc@5', 'val_acc@1', 'val_acc@5']
            )

    # make parallel
    model = torch.nn.DataParallel(model)

    # criterion for loss
    if CONFIG.class_weight:
        criterion = nn.CrossEntropyLoss(
            weight=get_class_weight(n_classes=CONFIG.n_classes).to(device)
        )
    else:
        criterion = nn.CrossEntropyLoss()

    # train and validate model
    print('\n------------------------Start training------------------------\n')
    train_losses = []
    val_losses = []
    train_top1_accuracy = []
    train_top5_accuracy = []
    val_top1_accuracy = []
    val_top5_accuracy = []

    for epoch in range(begin_epoch, CONFIG.max_epoch):

        # training
        train_loss, train_acc1, train_acc5 = train(
            train_loader, model, criterion, optimizer, epoch, CONFIG, device)

        train_losses.append(train_loss)
        train_top1_accuracy.append(train_acc1)
        train_top5_accuracy.append(train_acc5)

        # validation
        val_loss, val_acc1, val_acc5 = validate(
            val_loader, model, criterion, CONFIG, device)

        val_losses.append(val_loss)
        val_top1_accuracy.append(val_acc1)
        val_top5_accuracy.append(val_acc5)

        # scheduler
        if CONFIG.optimizer == 'SGD':
            scheduler.step(val_loss)

        # save a model if top1 acc is higher than ever
        # save base models, NOT DataParalled models
        if best_acc1 < val_top1_accuracy[-1]:
            best_acc1 = val_top1_accuracy[-1]
            torch.save(
                model.module.state_dict(),
                os.path.join(CONFIG.result_path, 'best_acc1_model.prm')
            )
        
        # save checkpoint every epoch
        save_checkpoint(
            CONFIG, epoch, model.module, optimizer, best_acc1, scheduler)

        # save a model every 10 epoch
        # save base models, NOT DataParalled models
        if epoch % 10 == 0 and epoch != 0:
            torch.save(
                model.module.state_dict(),
                os.path.join(
                    CONFIG.result_path, 'epoch_{}_model.prm'.format(epoch))
            )

        # tensorboardx
        if writer is not None:
            writer.add_scalars("loss", {
                'train': train_losses[-1],
                'val': val_losses[-1]
            }, epoch)
            writer.add_scalars("train_acc", {
                'top1': train_top1_accuracy[-1],
                'top5': train_top5_accuracy[-1]
            }, epoch)
            writer.add_scalars("val_acc", {
                'top1': val_top1_accuracy[-1],
                'top5': val_top5_accuracy[-1]
            }, epoch)

        # write logs to dataframe and csv file
        tmp = pd.Series([
            epoch,
            optimizer.param_groups[0]['lr'],
            train_losses[-1],
            val_losses[-1],
            train_top1_accuracy[-1],
            train_top5_accuracy[-1],
            val_top1_accuracy[-1],
            val_top5_accuracy[-1],
        ], index=log.columns)

        log = log.append(tmp, ignore_index=True)
        log.to_csv(os.path.join(CONFIG.result_path, 'log.csv'), index=False)

        print(
            'epoch: {}\tlr: {}\tloss train: {:.4f}\tloss val: {:.4f}\tval_acc1: {:.5f}\tval_acc5: {:.4f}'
            .format(epoch, optimizer.param_groups[0]['lr'], train_losses[-1],
                    val_losses[-1], val_top1_accuracy[-1], val_top5_accuracy[-1])
        )

    # save base models, NOT DataParalled models
    torch.save(
        model.module.state_dict(), os.path.join(CONFIG.result_path, 'final_model.prm'))


if __name__ == '__main__':
    main()
