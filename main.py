# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import wandb
import argparse
import datetime
from distutils.command.config import config
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler

import datasets
import util.misc as utils
from datasets import build_dataset, get_coco_api_from_dataset, get_api_from_dataset
from engine import evaluate, train_one_epoch
from models import build_model

import os


def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--lr_backbone', default=0.0001, type=float)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=20, type=int)
    parser.add_argument('--lr_drop', default=200, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')

    # Model parameters
    parser.add_argument('--pretrained_weights', type=str, default=None,#'/nobackup/yb/modelZoo/vistr_r50.pth',
                        help="Path to the pretrained model.")
    # * Backbone
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")

    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=384, type=int,
                        help="Size of the embeddings (dimension of the transformer)") #256->
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=100, type=int,
                        help="Number of query slots")
    parser.add_argument('--pre_norm', action='store_true')

    # * Segmentation -ignore
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")

    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")
    # * Matcher
    parser.add_argument('--set_cost_class', default=1, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")
    # * Loss coefficients
    parser.add_argument('--mask_loss_coef', default=1, type=float)
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--eos_coef', default=0.1, type=float,
                        help="Relative classification weight of the no-object class")

    # dataset parameters
    parser.add_argument('--dataset_file', default='argoverse')
    parser.add_argument('--dataset_path', type=str, default='/nobackup/yb/Argoverse')
    # parser.add_argument('--coco_panoptic_path', type=str)
    parser.add_argument('--remove_difficult', action='store_true')
    parser.add_argument('--num_frames', default=1, type=int,
                        help="Number of frames")
    parser.add_argument('--output_dir', default='/nobackup/yb/Exp/vptr',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=2, type=int)
    parser.add_argument('--current', dest='future', action='store_false')
    parser.add_argument('--no_data_augmentation', dest='data_augmentation', action='store_false')
    
    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    return parser


def main(args):
    utils.init_distributed_mode(args)
    # print("git:\n  {}\n".format(utils.get_sha()))

    # if args.frozen_weights is not None:
    #     assert args.masks, "Frozen training is meant for segmentation only"
    print(args)

    device = torch.device(args.device) #torch.device("cuda:{}".format(args.local_rank)) if args.local_rank else 

    # fix the seed for reproducibility #TODO: check the function of torch.backends
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    seed = args.seed# + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # * build model
    model, criterion, postprocessors = build_model(args)
    model.to(device)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],output_device=args.local_rank)
        model_without_ddp = model.module
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    if 'LOCAL_RANK' not in os.environ or int(os.environ['LOCAL_RANK']) == 0:
        wandb.watch(model_without_ddp,log='all')

    param_dicts = [
        {"params": [p for n, p in model_without_ddp.named_parameters() if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": args.lr_backbone,
        },
    ]
    # * make optimizer
    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                  weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)

    # * make dataloader
    dataset_train = build_dataset(image_set='train', args=args)
    dataset_val = build_dataset(image_set='val', args=args)

    # # sampled the dataset to speed up trainning for debuging
    # sampled_indices = random.sample(list(range(len(dataset_train))),len(dataset_train))[:500]
    # dataset_train = torch.utils.data.Subset(dataset_train,sampled_indices)

    # subsampling the dataset
    if args.distributed:
        sampler_train = DistributedSampler(dataset_train) #TODO: sampler that restricts data loading to a subset of the dataset.
        sampler_val = DistributedSampler(dataset_val, shuffle=False)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, args.batch_size, drop_last=True)

    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                   collate_fn=utils.collate_fn, num_workers=args.num_workers)
    data_loader_val = DataLoader(dataset_val, args.batch_size, sampler=sampler_val,
                                 drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers)
    # * get api for evaluation
    if args.dataset_file == "coco_panoptic":
        # We also evaluate AP during panoptic training, on original coco DS
        coco_val = datasets.coco.build("val", args)
        base_ds = get_coco_api_from_dataset(coco_val)
    elif args.dataset_file == "coco":
        base_ds = get_coco_api_from_dataset(dataset_val)
    else:
        base_ds = get_api_from_dataset(dataset_val)
        train_base_ds = get_api_from_dataset(dataset_train)

    # if args.frozen_weights is not None:
    #     checkpoint = torch.load(args.frozen_weights, map_location='cpu')
    #     model_without_ddp.detr.load_state_dict(checkpoint['model'])
    
    # *load model
    if args.pretrained_weights is not None:
        checkpoint = torch.load(args.pretrained_weights, map_location='cpu')['model']
        # del checkpoint["vistr.class_embed.weight"] #TODO: figure out why del
        # del checkpoint["vistr.class_embed.bias"]
        # del checkpoint["vistr.query_embed.weight"]

        if args.distributed:
            model.module.load_state_dict(checkpoint,strict=False) #Once you wrap the model with nn.DataParallel you get an "extra" .module in your way.
        else:
            model.load_state_dict(checkpoint,strict=False)

    output_dir = Path(args.output_dir)
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1

    if args.eval:
        test_stats, coco_evaluator = evaluate(model, criterion, postprocessors,
                                              data_loader_val, base_ds, device, args.output_dir)
        if args.output_dir:
            utils.save_on_master(coco_evaluator.coco_eval["bbox"].eval, output_dir / "eval.pth")
        return

    print("Start training")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            sampler_train.set_epoch(epoch)

        train_stats = train_one_epoch(
            model, criterion, postprocessors, data_loader_train, optimizer, device, epoch,
            args.clip_max_norm)
        
        lr_scheduler.step()

        evaluate(
            model, criterion, postprocessors, data_loader_train, train_base_ds, device, args.output_dir, mode='train_eval'
        )
        # save every epoch
        if args.output_dir:
            checkpoint_paths = [output_dir / 'checkpoint.pth']
            # extra checkpoint before LR drop and every 100 epochs
            if (epoch + 1) % args.lr_drop == 0 or (epoch + 1) % 100 == 0:
                checkpoint_paths.append(output_dir / f'checkpoint{epoch:04}.pth')
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }, checkpoint_path)

        test_stats, coco_evaluator = evaluate(
            model, criterion, postprocessors, data_loader_val, base_ds, device, args.output_dir
        )

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'test_{k}': v for k, v in test_stats.items()},
                     'epoch': epoch,
                     'n_parameters': n_parameters}

        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

            # for evaluation logs
            if coco_evaluator is not None:
                (output_dir / 'eval').mkdir(exist_ok=True)
                if "bbox" in coco_evaluator.coco_eval:
                    filenames = ['latest.pth']
                    if epoch % 50 == 0:
                        filenames.append(f'{epoch:03}.pth')
                    for name in filenames:
                        torch.save(coco_evaluator.coco_eval["bbox"].eval,
                                   output_dir / "eval" / name)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    # os.environ["CUDA_VISIBLE_DEVICES"] = '2,3'
    
    if 'LOCAL_RANK' not in os.environ or int(os.environ['LOCAL_RANK']) == 0:
        world_size = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1
        wandb.init(
            project='one_frame_vptr',
            config={'learning_rate:':args.lr,
                    'batch_size_per_gpu':args.batch_size,
                    'world_size': world_size,
                    'batch_size': world_size * args.batch_size,
                    'epochs':args.epochs,
                    'backbone':args.backbone,
                    'pre_trained':args.pretrained_weights,
                    'num_frames':args.num_frames,
                    'dataset':args.dataset_file,
                    'data_augmentation':args.data_augmentation
            })

    main(args)
