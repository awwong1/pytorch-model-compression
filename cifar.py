#!/usr/bin/env python3
"""CIFAR-10 and CIFAR-100 helper script
"""

from __future__ import print_function

import inspect
import logging
import os
import random
import torch
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms

from argparse import ArgumentParser
from torch.utils.data import DataLoader

MODEL_ARCHS = {name: value for name, value in inspect.getmembers(
    models) if inspect.isfunction(value) or inspect.ismodule(value)}
USE_CUDA = torch.cuda.is_available()


def main(args):
    # Preliminary Setup
    if USE_CUDA:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    if args.manual_seed is not None:
        random.seed(args.manual_seed)
        torch.manual_seed(args.manual_seed)
        if USE_CUDA:
            torch.cuda.manual_seed_all(args.manual_seed)
    logging.basicConfig(level=args.verbosity, format="%(message)s")
    # format="%(asctime)s %(levelname)s %(message)s"

    # Data
    logging.info("Preparing dataset %(dataset)s", {"dataset": args.dataset})
    train_transforms = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # https://github.com/kuangliu/pytorch-cifar/issues/19#issue-268972488
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])
    test_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])
    if args.dataset == "cifar10":
        data_class = datasets.CIFAR10
        num_classes = 10
    elif args.dataset == "cifar100":
        data_class = datasets.CIFAR100
        num_classes = 100
    else:
        raise NotImplementedError("{} Not implemented".format(args.dataset))

    trainset = data_class(root='./data', train=True,
                          download=True, transform=train_transforms)
    trainloader = DataLoader(
        trainset, batch_size=args.train_batch, shuffle=True, num_workers=args.workers)
    testset = data_class(root='./data', train=False,
                         download=False, transform=test_transforms)
    testloader = DataLoader(
        testset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)

    # Model & Architecture
    logging.info("Initializing model architecture \"%(arch)s\"",
                 {"arch": args.arch})
    model = MODEL_ARCHS.get(args.arch)(
        pretrained=False, num_classes=num_classes)
    logging.debug("%(model)s", {"model": model})

    model = torch.nn.DataParallel(model)
    if USE_CUDA:
        model = model.cuda()
        torch.backends.cudnn.benchmark = True

    num_params = sum([p.numel() for p in model.parameters()])
    num_learnable = sum([p.numel()
                         for p in model.parameters() if p.requires_grad])

    logging.info("Number of parameters: %(params)d (%(learnable)d learnable)", {
                 "params": num_params, "learnable": num_learnable})


def parse_arguments():
    """Parse and return the command line arguments
    """
    parser = ArgumentParser("CIFAR-10/100 Training")
    _verbosity = "INFO"
    parser.add_argument("-v", "--verbosity", type=str, choices=logging._nameToLevel.keys(), default=_verbosity, metavar="VERBOSITY",
                        help="output verbosity: {} (default: {})".format(" | ".join(logging._nameToLevel.keys()), _verbosity))
    parser.add_argument("--manual-seed", type=int, help="manual seed integer")
    parser.add_argument("-e", "--evaluate", action="store_true",
                        help="evaluate model on validation data")
    parser.add_argument("--gpu-id", default="0", type=str,
                        help="id(s) for CUDA_VISIBLE_DEVICES")

    # Dataset Options
    d_op = parser.add_argument_group("Dataset")
    d_op.add_argument("-d", "--dataset", default="cifar10",
                      type=str, choices=("cifar10", "cifar100"))
    avail_cpus = len(os.sched_getaffinity(0))
    d_op.add_argument("-w", "--workers", default=avail_cpus, type=int, metavar="N",
                      help=f"number of data-loader workers (default: {avail_cpus})")

    # Architecture Options
    a_op = parser.add_argument_group("Architectures")
    _architecture = "alexnet"
    a_op.add_argument("-a", "--arch", metavar="ARCH", default=_architecture,
                      choices=MODEL_ARCHS.keys(),
                      help="model architecture: {} (default: {})".format(" | ".join(MODEL_ARCHS.keys()), _architecture))
    # a_op.add_argument("--depth", type=int, default=29, help="Model depth.")
    # a_op.add_argument("--block-name", type=str, default="BasicBlock",
    #                   help="the building block for Resnet and Preresnet: BasicBlock, Bottleneck (default: Basicblock for cifar10/cifar100)")
    # a_op.add_argument("--cardinality", type=int, default=8,
    #                   help="Model cardinality (group).")
    # a_op.add_argument("--widen-factor", type=int, default=4,
    #                   help="Widen factor. 4 -> 64, 8 -> 128, ...")
    # a_op.add_argument("--growthRate", type=int, default=12,
    #                   help="Growth rate for DenseNet.")
    # a_op.add_argument("--compressionRate", type=int, default=2,
    #                   help="Compression Rate (theta) for DenseNet.")

    # Optimization Options
    o_op = parser.add_argument_group("Optimizations")
    _epochs = 300
    o_op.add_argument("--epochs", default=_epochs, type=int, metavar="N",
                      help=f"number of epochs to run (default: {_epochs})")
    _epoch_start = 0
    o_op.add_argument("--start-epoch", default=_epoch_start, type=int, metavar="N",
                      help=f"epoch start number (default: {_epoch_start})")
    _train_batch = 128
    o_op.add_argument("--train-batch", default=_train_batch, type=int, metavar="N",
                      help=f"train batchsize (default: {_train_batch})")
    _test_batch = 100
    o_op.add_argument("--test-batch", default=_test_batch, type=int, metavar="N",
                      help=f"test batchsize (default: {_test_batch})")
    _lr = 0.1
    o_op.add_argument("--lr", "--learning-rate", default=_lr, type=float, metavar="LR",
                      help=f"initial learning rate (default: {_lr})")
    _dropout = 0
    o_op.add_argument("--drop", "--dropout", default=_dropout, type=float, metavar="Dropout",
                      help=f"Dropout ratio (default: {_dropout})")
    _schedule = [150, 225]
    o_op.add_argument("--schedule", type=int, nargs="+", default=_schedule,
                      help=f"Decrease LR at these epochs (default: {_schedule})")
    _gamma = 0.1
    o_op.add_argument("--gamma", type=float, default=_gamma,
                      help=f"LR is multiplied by gamma on schedule (default: {_gamma})")
    _momentum = 0.9
    o_op.add_argument("--momentum", default=_momentum, type=float, metavar="M",
                      help=f"momentum (default: {_momentum})")
    _wd = 5e-4
    o_op.add_argument("--weight-decay", "--wd", default=_wd, type=float, metavar="W",
                      help=f"weight decay (default: {_wd})")

    # Checkpoint Options
    c_op = parser.add_argument_group("Checkpoints")
    _checkpoint = "checkpoint"
    c_op.add_argument("-c", "--checkpoint", default=_checkpoint, type=str, metavar="PATH",
                      help=f"path to save checkpoint (default: {_checkpoint})")
    _resume = None
    c_op.add_argument("--resume", default=_resume, type=str, metavar="PATH",
                      help=f"path to latest checkpoint (default: {_resume})")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
