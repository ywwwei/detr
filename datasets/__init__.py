# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch.utils.data
import torchvision

from .coco import build as build_coco
# from .ytvos import build as build_ytvis
from .argoversehd import build_argoverse, build_ytvis_argoformat


def get_coco_api_from_dataset(dataset):
    for _ in range(10):
        # if isinstance(dataset, torchvision.datasets.CocoDetection):
        #     break
        if isinstance(dataset, torch.utils.data.Subset):
            dataset = dataset.dataset
    if isinstance(dataset, torchvision.datasets.CocoDetection):
        return dataset.coco

def get_api_from_dataset(dataset):
    return dataset.api

def build_dataset(image_set, args):
    if args.dataset_file == 'coco':
        return build_coco(image_set, args)
    if args.dataset_file == 'coco_panoptic':
        # to avoid making panopticapi required for coco
        from .coco_panoptic import build as build_coco_panoptic
        return build_coco_panoptic(image_set, args)
    # if args.dataset_file == 'ytvis':
    #     return build_ytvis(image_set, args)
    if args.dataset_file == 'argoverse':
        return build_argoverse(image_set, args)
    if args.dataset_file == 'ytvis_argoformat':
        return build_ytvis_argoformat(image_set, args)
    raise ValueError(f'dataset {args.dataset_file} not supported')
