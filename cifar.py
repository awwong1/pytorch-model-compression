#!/usr/bin/env python3
"""CIFAR-10 and CIFAR-100 helper script
"""

from __future__ import print_function

import inspect
import logging
import os
import random
import shutil
import torch
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
from argparse import ArgumentParser
from pprint import pformat
from progress.bar import Bar
from time import time
from torch.utils.data import DataLoader

from utils import AverageMeter, Scribe, accuracy

MODEL_ARCHS = {name: value for name, value in inspect.getmembers(
    models) if inspect.isfunction(value) or inspect.ismodule(value)}
USE_CUDA = torch.cuda.is_available()


def main(**args):
    # Preliminary Setup
    if USE_CUDA:
        os.environ["CUDA_VISIBLE_DEVICES"] = args["gpu_id"]
    if args["manual_seed"] is not None:
        random.seed(args["manual_seed"])
        torch.manual_seed(args["manual_seed"])
        if USE_CUDA:
            torch.cuda.manual_seed_all(args["manual_seed"])
    if not os.path.isdir(args["checkpoint"]):
        os.makedirs(args["checkpoint"], exist_ok=True)

    logging.basicConfig(level=args["verbosity"], format="%(message)s")
    # format="%(asctime)s %(levelname)s %(message)s"

    logging.info("• Options: %s", pformat(args))

    # Data
    logging.info("• Preparing '%(dataset)s' dataset", args)
    base_transforms = [
        transforms.ToTensor(),
        # https://github.com/kuangliu/pytorch-cifar/issues/19#issue-268972488
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ]
    train_transforms = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
    ] + base_transforms)
    test_transforms = transforms.Compose(base_transforms)
    if args["dataset"] == "cifar10":
        data_class = datasets.CIFAR10
        num_classes = 10
    elif args["dataset"] == "cifar100":
        data_class = datasets.CIFAR100
        num_classes = 100
    else:
        assert args["dataset"] in (
            "cifar10", "cifar100"), "Unsupported dataset: " + args["dataset"]

    trainset = data_class(root="./data", train=True,
                          download=True, transform=train_transforms)
    trainloader = DataLoader(
        trainset, batch_size=args["train_batch"], shuffle=True, num_workers=args["workers"])
    testset = data_class(root="./data", train=False,
                         download=False, transform=test_transforms)
    testloader = DataLoader(
        testset, batch_size=args["test_batch"], shuffle=False, num_workers=args["workers"])

    # Model & Architecture
    logging.info("• Initializing '%(arch)s' architecture", args)
    # todo: don"t use imagenet defaults, reduce FC layers
    model = MODEL_ARCHS.get(args["arch"])(
        pretrained=False, num_classes=num_classes)
    logging.debug("%(model)s", {"model": model})

    model = torch.nn.DataParallel(model)
    if USE_CUDA:
        model = model.cuda()
        torch.backends.cudnn.benchmark = True

    num_params = sum([p.numel() for p in model.parameters()])
    num_learnable = sum([p.numel()
                         for p in model.parameters() if p.requires_grad])

    logging.info("• Number of parameters: %(params)d (%(learnable)d learnable)", {
                 "params": num_params, "learnable": num_learnable})

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args["lr"], momentum=args["momentum"], weight_decay=args["weight_decay"])

    # Resume & Initialize Progress Logger
    best_acc = 0
    start_epoch = args["start_epoch"]
    title = args["dataset"] + "-" + args["arch"]
    if args["resume"]:
        logging.info("• Loading from checkpoint")
        assert os.path.isfile(
            args["resume"]), "Invalid checkpoint path: " + args["resume"]
        args["checkpoint"] = os.path.dirname(args["resume"])
        checkpoint = torch.load(args["resume"])
        best_acc = checkpoint["best_acc"]
        start_epoch = checkpoint["epoch"]
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scribe = Scribe(os.path.join(args["checkpoint"],
                                     "progress.txt"), title=title, resume=True)
    else:
        scribe = Scribe(os.path.join(args["checkpoint"],
                                     "progress.txt"), title=title)
        scribe.set_names(['Learning Rate', 'Train Loss',
                          'Valid Loss', 'Train Acc.', 'Valid Acc.'])

    if args["mode"] == "evaluate":
        logging.info("Only evaluation")
        with torch.no_grad():
            test_loss, test_acc = test(
                testloader, model, criterion, start_epoch)
        logging.info('Test Loss:  %(loss).8f, Test Acc:  %(acc).2f', {
                     "loss": test_loss, "acc": test_acc})
    elif args["mode"] == "train":
        lr = args["lr"]
        for epoch in range(start_epoch, args["epochs"]):
            lr = update_learning_rate(
                lr, args["schedule"], args["gamma"], optimizer, epoch)
            logging.info("Epoch %(cur_epoch)d/%(epochs)d | LR: %(lr)f",
                         {"cur_epoch": epoch + 1, "epochs": args["epochs"], "lr": lr})

            train_loss, train_acc = train(
                trainloader, model, criterion, optimizer, epoch)
            with torch.no_grad():
                test_loss, test_acc = test(testloader, model, criterion, epoch)

            # append model progress
            scribe.append((lr, train_loss, test_loss, train_acc, test_acc))

            # save the model
            is_best = test_acc > best_acc
            best_acc = max(test_acc, best_acc)
            save_checkpoint({
                "epoch": epoch + 1,
                "state_dict": model.state_dict(),
                "acc": test_acc,
                "best_acc": best_acc,
                "optimizer": optimizer.state_dict()
            }, is_best, checkpoint=args["checkpoint"])


def train(trainloader, model, criterion, optimizer, epoch):
    return _pass("Train", trainloader, model, criterion, optimizer, epoch)


def test(testloader, model, criterion, epoch):
    return _pass("Test", testloader, model, criterion, None, epoch)


def _pass(mode, dataloader, model, criterion, optimizer, epoch):
    """Perform one train or test pass through the data
    """
    batch_time = AverageMeter("Batch Time")
    data_time = AverageMeter("Data Time")
    losses = AverageMeter("Losses")
    top1 = AverageMeter("Top 1 Accuracy")
    top5 = AverageMeter("Top 5 Accuracy")
    end = time()

    if mode == "Train":
        model.train()
    elif mode == "Test":
        model.eval()
    else:
        assert mode in ("Train", "Test"), f"Unsupported mode {mode}"

    bar = Bar(mode, max=len(dataloader))
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        # measure data loading time
        data_time.update(time() - end)
        if USE_CUDA:
            inputs, targets = inputs.cuda(), targets.cuda()

        # compute output
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # measure accuracy and record loss
        # pylint: disable=unbalanced-tuple-unpacking
        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        losses.update(loss.data.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))

        if mode == "Train":
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # measure elapsed time
        batch_time.update(time() - end)
        end = time()

        # plot progress
        bar.suffix = "({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}".format(
            batch=batch_idx + 1,
            size=len(dataloader),
            data=data_time.avg,
            bt=batch_time.avg,
            total=bar.elapsed_td,
            eta=bar.eta_td,
            loss=losses.avg,
            top1=top1.avg,
            top5=top5.avg,
        )
        bar.next()
    bar.finish()
    return (losses.avg, top1.avg)


def update_learning_rate(lr, schedule, gamma, optimizer, epoch):
    if epoch in schedule:
        _lr = lr * gamma
        logging.info("lr scheduled update from %(lr)f to %(_lr)f on epoch %(epoch)d", {
                     "lr": lr, "_lr": _lr, "epoch": epoch})
        for param_group in optimizer.param_groups:
            param_group["lr"] = _lr
        lr = _lr
    return lr


def save_checkpoint(state, is_best, checkpoint='checkpoint', filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(
            checkpoint, 'model_best.pth.tar'))


def parse_arguments():
    """Parse and return the command line arguments
    """
    parser = ArgumentParser("CIFAR-10/100 Training")
    _verbosity = "INFO"
    parser.add_argument("-v", "--verbosity", type=str, choices=logging._nameToLevel.keys(), default=_verbosity, metavar="VERBOSITY",
                        help="output verbosity: {} (default: {})".format(" | ".join(logging._nameToLevel.keys()), _verbosity))
    parser.add_argument("--manual-seed", type=int, help="manual seed integer")
    _mode = "train"
    parser.add_argument("-m", "--mode", type=str, default=_mode, choices=["train", "evaluate"],
                        help=f"script execution mode (default: {_mode})")
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

    return vars(parser.parse_args())


if __name__ == "__main__":
    options = parse_arguments()
    main(**options)
