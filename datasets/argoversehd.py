"""
argoverseHD data loader
"""
from pathlib import Path

import torch
import torch.utils.data
import torchvision
from .apis import argoverseAPI as ann_API
from . import transforms as T
import os
from PIL import Image
from random import randint
import cv2
import random

class ArgoverseHDDataset:
    def __init__(
        self, 
        data_dir, 
        json_file, 
        transforms = None, 
        num_frames = 3, 
        future = False
    ):

        self.data_dir = data_dir # folder of video folder containing sequences
        self.json_file = json_file # train.json or test.json
        self._transforms = transforms
        self.num_frames = num_frames

        self.api = ann_API(json_file) # annotations

        target_im_ids = []
        input_seq_ids = []

        for sid, im_ids in self.api.sidToImgs.items():
            vid_len = len(im_ids)
            for fid in range(len(im_ids)): 
                # target img annotation
                target_im_ids.append(im_ids[fid])
                # input clip
                inds = list(range(self.num_frames))
                inds = [i%vid_len for i in inds][::-1]
                if future:
                    seq_ids_ = [im_ids[fid-i-1] for i in inds]
                else:
                    seq_ids_ = [im_ids[fid-i] for i in inds]
                input_seq_ids.append(seq_ids_)
                
        assert len(target_im_ids) == len(input_seq_ids)

        self.target_im_ids = target_im_ids # a list. each element is the target future im id
        self.input_seq_ids = input_seq_ids # a list. each element is the list of input history frames with length num_frames

    def __len__(self):
        return len(self.target_im_ids)
    
    def load_images(self, idx):
        input_im_ids = self.input_seq_ids[idx]
        imgs = []
        for im_id in input_im_ids:
            image = self.api.imgs[im_id]
            if "ytvos_data" in str(self.data_dir):
                img_path = self.data_dir/image['name']
            else: #Argoverse
                img_path = self.data_dir/self.api.sequences[image['sid']]/'ring_front_center'/image['name']
            imgs.append(Image.open(img_path).convert('RGB'))
        return imgs
    
    def load_anns(self,idx):
        target_im_id = self.target_im_ids[idx]
        return self.api.imgToAnns[target_im_id]

    def __getitem__(self, idx):
        target_im_id = self.target_im_ids[idx]
        target_annos = self.api.imgToAnns[target_im_id]

        target = {}
        # return torch.cat(img,dim=0), target
        # anno = [obj for obj in anno if 'iscrowd' not in obj or obj['iscrowd'] == 0]
        boxes = [anno["bbox"] for anno in target_annos]
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2] # convert [x,y,w,h] to [x0,y0,x1,y1]
        # boxes[:, 0::2].clamp_(min=0, max=w)
        # boxes[:, 1::2].clamp_(min=0, max=h)

        classes = [anno["category_id"] for anno in target_annos]
        classes = torch.tensor(classes, dtype=torch.int64)

        image_id = torch.tensor([target_im_id])

        area = torch.tensor([anno["area"] for anno in target_annos])
        iscrowd = torch.tensor([anno["iscrowd"] if "iscrowd" in anno else 0 for anno in target_annos])

        target["boxes"] = boxes
        target["labels"] = classes
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        imgs = self.load_images(idx) # a list of PIL Image
        w, h = imgs[0].size

        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])

        if self._transforms is not None:
            img, target = self._transforms(imgs, target)

        return  torch.cat(img,dim=0), target #return video ( [(C,H,W)]*num_frames -> (C*num_frame,H,W)), target

def make_coco_transforms(image_set):

    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]

    if image_set == 'train':
        return T.Compose([
            T.RandomHorizontalFlip(),
            # T.PhotometricDistort(),
            T.RandomSelect(
                T.RandomResize(scales, max_size=1333),
                T.Compose([
                    T.RandomResize([400, 500, 600]),
                    T.RandomSizeCrop(384, 600),
                    T.RandomResize(scales, max_size=1333),
                ])
            ),
            # To suit the GPU memory the scale might be different
            T.RandomResize([300], max_size=540),#for r50
            #T.RandomResize([280], max_size=504),#for r101
            normalize,
        ])

    if image_set == 'val':
        return T.Compose([
            T.RandomResize([360], max_size=640),
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')


def no_transforms(image_set):

    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]) # TODO:

    # scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768]

    if image_set == 'train':
        return T.Compose([
            # T.RandomHorizontalFlip(),
            # T.RandomResize(scales, max_size=800),
            # T.PhotometricDistort(),
            T.Compose([
                    #  T.RandomResize([400, 500, 600]),
                    #  T.RandomSizeCrop(384, 600),
                     # To suit the GPU memory the scale might be different
                     T.RandomResize([300], max_size=540),#for r50
                     #T.RandomResize([280], max_size=504),#for r101
            ]),
            normalize,
        ])
        # return T.Compose([T.RandomResize([500], max_size=540), T.ToTensor()])

    if image_set == 'val':
        return T.Compose([
            T.RandomResize([360], max_size=640),
            normalize,
        ])
        # return T.Compose([T.RandomResize([500], max_size=540), T.ToTensor()])

    raise ValueError(f'unknown {image_set}')

def build_argoverse(image_set, args):
    """
    image_set: 'train' or 'val'
    """
    root = Path(args.dataset_path)
    assert root.exists(), f'provided dataset path {root} does not exist'
    # mode = 'instances'
    PATHS = {
            "train": (root / "Argoverse-1.1/tracking" / "train", root / "Argoverse-HD/annotations" / "train.json"),
            "val": (root / "Argoverse-1.1/tracking" / "val", root / "Argoverse-HD/annotations" / "val.json")
    }

    data_dir, json_file = PATHS[image_set]
    dataset = ArgoverseHDDataset(data_dir, json_file, transforms=make_coco_transforms(image_set), num_frames = args.num_frames, future=args.future)
    print(image_set, ' ',dataset.__len__(), ' measurements')
    return dataset

def build_ytvis_argoformat(image_set, args):
    root = Path(args.dataset_path)
    assert root.exists(), f'provided YTVOS path {root} does not exist'
    PATHS = {
        "train": (root / "train/JPEGImages", root / "annotations" / 'train_argoformat.json'),
        "val": (root / "train/JPEGImages", root / "annotations" / 'valid_argoformat.json'),
    }
    img_folder, ann_file = PATHS[image_set]
    dataset = ArgoverseHDDataset(img_folder, ann_file, transforms=make_coco_transforms(image_set), num_frames = args.num_frames, future=args.future)
    print(image_set, len(dataset), ' measurements')
    return dataset

    