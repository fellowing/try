import csv
import numpy as np
import argparse
import os
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
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler

import pickle
from data_loader import MRIData_allbatch, MRIData_onebatch
from cnn_lstm import Conv, CNNBlock
from torch.utils.data import DataLoader 
from torch.nn.parallel import DistributedDataParallel

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

# 第一个参数是 flag -a，可以在命令好调用的时候使用这个缩写
# 第二个参数是 name 可以在命令行调用(也可以在 py 文件中调用，但是如果有 - 要改成 _)
# metavar - 在 usage 说明中的参数名称，对于必选参数默认就是参数名称，对于可选参数默认是全大写的参数名称.
# default - 不指定参数时的默认值
# dest - 解析后的参数名称，默认情况下，对于可选参数选取最长的名称，中划线转换为下划线
# help - 参数的帮助信息，当指定为 argparse.SUPPRESS 时表示不显示该参数的帮助信息.
# default - 不指定参数时的默认值

# 显卡数量
parser.add_argument('-j',
                    '--workers',
                    default=4,
                    type=int,
                    metavar='N',
                    help='number of data loading workers (default: 4)')

# 训练次数
parser.add_argument('--epochs',
                    default=1000,
                    type=int,
                    metavar='N',
                    help='number of total epochs to run')

# 开始训练的 epoch
parser.add_argument('--start-epoch',
                    default=0,
                    type=int,
                    metavar='N',
                    help='manual epoch number (useful on restarts)')

# 训练的 batch size
parser.add_argument('-b',
                    '--batch-size',
                    default=10,
                    type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                    'batch size of all GPUs on the current node when '
                    'using Data Parallel or Distributed Data Parallel')

# learn rate
parser.add_argument('--lr',
                    '--learning-rate',
                    default=0.01,
                    type=float,
                    metavar='LR',
                    help='initial learning rate',
                    dest='lr')

# 设置随机梯度下降法的 momentum 可以加快训练
parser.add_argument('--momentum',
                    default=0.9,
                    type=float,
                    metavar='M',
                    help='momentum')

# 设置随机梯度下降法的 weight_decay 可以防止过拟合
parser.add_argument('--wd',
                    '--weight-decay',
                    default=1e-4,
                    type=float,
                    metavar='W',
                    help='weight decay (default: 1e-4)',
                    dest='weight_decay')

# 定义打印的时候 i 的大小
parser.add_argument('-p',
                    '--print-freq',
                    default=10,
                    type=int,
                    metavar='N',
                    help='print frequency (default: 10)')

# 定义是否进行 evaluate，python try2.py -e 则传递参数为 True，不调用则为 F
parser.add_argument('-e',
                    '--evaluate',
                    dest='evaluate',
                    action='store_true',
                    help='evaluate model on validation set')

# 定义是否使用与训练模型，
parser.add_argument('--pretrained',
                    dest='pretrained',
                    action='store_true',
                    help='use pre-trained model')

# 定义种子
parser.add_argument('--seed',
                    default=22716,
                    type=int,
                    help='seed for initializing training. ')

# 定义训练集和测试集文件路径
parser.add_argument('--data',
                    default=['./data/train_data.pkl','./data/test_data.pkl'],
                    type=list,
                    help='path of train and test set path')

# 增加 modal 的参数
parser.add_argument('--modal',
                    default='ADC',
                    type=str,
                    help='the modal want to use')

# 增加最大图像数量
parser.add_argument('--max_num',
                    default=3,
                    type=int,
                    help='the max num of MRI used in one modals')

# 添加默认影像大小参数
parser.add_argument('--std_dim',
                    default={'ADC': (160,160,20), 'T1WI+C': (320, 320, 20), 
                    'T1WI': (640, 640, 20), 'CBV': (220, 220, 21),
                    'T2WI': (640, 640, 20), 'T2_FLAIR': (320, 320, 20), 
                    'T1_MPRAGE': (256, 256, 192)},
                    type=dict,
                    help='the std dim of every modal mris')

# 修改后，同步各 GPU 中数据切片的统计信息，用于分布式的 evaluation
def reduce_mean(tensor, nprocs):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= nprocs
    return rt

# 定义 main 函数
def main():
    args = parser.parse_args()
    args.nprocs = torch.cuda.device_count()

    mp.spawn(main_worker, nprocs=args.nprocs, args=(args.nprocs, args))

# 定义一个设置种子的辅助函数，来自
def wif(id):
    process_seed = torch.initial_seed()
    # Back out the base_seed so we can use all the bits.
    base_seed = process_seed - id
    ss = np.random.SeedSequence([id, base_seed])
    # More than 128 bits (4 32-bit words) would be overkill.
    np.random.seed(ss.generate_state(4))

# 
def main_worker(local_rank, nprocs, args):
    args.local_rank = local_rank

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.backends.cudnn.benchmark = False
        np.random.seed(args.seed)
        cudnn.deterministic = True
        torch.backends.cudnn.deterministic = True
        torch.cuda.manual_seed_all(args.seed)

        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    # best_acc1 = .0

    dist.init_process_group(backend='nccl',
                            init_method='tcp://127.0.0.1:23456',
                            world_size=args.nprocs,
                            rank=local_rank)
    
    # backend str/Backend 是通信所用的后端，可以是"ncll" "gloo"或者是一个
    # init_method 这个URL指定了如何初始化互相通信的进程
    # world_size int 执行训练的所有的进程数
    # rank int this进程的编号，也是其优先级

    # create model
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
        # model = models.__dict__[args.arch](pretrained=True)
    else:
        # print("=> creating model '{}'".format(args.arch))
        model = Conv(CNNBlock,[2,2,None,None])

    torch.cuda.set_device(local_rank)
    model.cuda(local_rank)
    # When using a single GPU per process and per
    # DistributedDataParallel, we need to divide the batch size
    # ourselves based on the total number of GPUs we have
    args.batch_size = int(args.batch_size / args.nprocs)
    model = DistributedDataParallel(model, device_ids=[local_rank])

    # define loss function (criterion) and optimizer
    criterion = nn.BCELoss().cuda(local_rank)

    optimizer = torch.optim.SGD(model.parameters(),
                                args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    cudnn.benchmark = True

    # Data loading code
    train_data_path = pickle.load(open(args.data[0], "rb"))

    train_dataset = MRIData_allbatch(train_data_path, args.std_dims[args.modal], 
                                    args.max_num, args.modal)
    train_sampler = DistributedSampler(train_dataset)
    train_loader = DataLoader(train_dataset,
                            batch_size=args.batch_size,
                            num_workers=2,
                            # pin_memory=True,
                            sampler=train_sampler,
                            # worker_init_fn = wif
                            )
    
    # 如果使用 pin_memory = True 则使用锁业内存，速度快但是可能爆内存
    # worker_init_fn=wif 是 numpy 作者建议的用于每个 epoch 产生随机数据，
    # 但是测试相同的代码，相同的 epoch 不用也会有相同的结果，建议暂时不用

    val_data_path = pickle.load(open(args.data[1], "rb"))
    val_dataset = MRIData_allbatch(val_data_path, args.std_dims[args.modal], 
                                    args.max_num, args.modal)
    val_sampler = DistributedSampler(val_dataset)
    val_loader = DataLoader(val_dataset,
                            shuffle = True,
                            batch_size=args.batch_size,
                            num_workers=2,
                            # pin_memory=True,
                            sampler=val_sampler,
                            # worker_init_fn = wif
                            )

    if args.evaluate:
        validate(val_loader, model, criterion, local_rank, args)
        return

    for epoch in range(args.start_epoch, args.epochs):

        train_sampler.set_epoch(epoch)
        val_sampler.set_epoch(epoch)

        adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, local_rank,
              args)

        # evaluate on validation set
        acc1 = validate(val_loader, model, criterion, local_rank, args)

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        if args.local_rank == 0:
            save_checkpoint(
                {
                    'epoch': epoch + 1,
                    'arch': args.arch,
                    'state_dict': model.module.state_dict(),
                    'best_acc1': best_acc1,
                }, is_best)


def train(train_loader, model, criterion, optimizer, epoch, local_rank, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(len(train_loader),
                             [batch_time, data_time, losses, top1, top5],
                             prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        images = images.cuda(local_rank, non_blocking=True)
        target = target.cuda(local_rank, non_blocking=True)

        # compute output
        output = model(images)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        # 在 reduce 之前插入了一个同步 API
        torch.distributed.barrier()

        reduced_loss = reduce_mean(loss, args.nprocs)
        reduced_acc1 = reduce_mean(acc1, args.nprocs)
        reduced_acc5 = reduce_mean(acc5, args.nprocs)

        losses.update(reduced_loss.item(), images.size(0))
        top1.update(reduced_acc1.item(), images.size(0))
        top5.update(reduced_acc5.item(), images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)


def validate(val_loader, model, criterion, local_rank, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    progress = ProgressMeter(len(val_loader), [batch_time, losses, top1],
                             prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            # 将图像和标签都加载到显卡上
            images = images.cuda(local_rank, non_blocking=True)
            target = target.cuda(local_rank, non_blocking=True)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1 = accuracy(output, target, topk=(1,))

            torch.distributed.barrier()

            reduced_loss = reduce_mean(loss, args.nprocs)
            reduced_acc1 = reduce_mean(acc1, args.nprocs)


            losses.update(reduced_loss.item(), images.size(0))
            top1.update(reduced_acc1.item(), images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        # TODO: this should also be done with the ProgressMeter
        print(' * Acc@1 {top1.avg:.3f}'.format(top1=top1))

    return top1.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


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


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1**(epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1, )):
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


if __name__ == '__main__':
    main()

