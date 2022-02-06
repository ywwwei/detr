"""
youtube-vis data loader
"""
from pathlib import Path

import torch
import torch.utils.data
import torchvision
from .apis import youtubevisAPI as ann_API
from . import transforms as T
import os
from PIL import Image
from random import randint
import cv2
import random

class YTVISDataset:
    def __init__(
        self, 
        data_dir, 
        json_file, 
        transforms = None, 
        return_masks = False, 
        num_frames = 3, 
        step = None, 
        future = True,
    ):

        self.data_dir = data_dir # folder of video folder containing sequences
        self.json_file = json_file # train.json or test.json
        self._transforms = transforms
        self.return_masks = return_masks
        self.num_frames = num_frames
        if step is None:
            step = self.num_frames+1
        self.step = step # step between each target t2-t1

        self.api = ann_API(json_file) # annotations

        target_im_ids = []
        input_seq_ids = []

        for sid in range(len(self.api.sequences)):
            im_ids =  self.api.sidToImgs[sid] # im_ids in the sequence of current sid
            for fid in range(self.num_frames,len(im_ids),step): 
                # skip first num_frame frames and last extra frames
                if future:
                    target_im_ids.append(im_ids[fid])
                else:
                    target_im_ids.append(im_ids[fid-1]) # the last input as prediction

                seq_ids_ = [im_ids[fid-i] for i in range(self.num_frames,0,-1)]
                input_seq_ids.append(seq_ids_)
        assert len(target_im_ids) == len(input_seq_ids)

        self.target_im_ids = target_im_ids # a list. each element is the target future im id
        self.input_seq_ids = input_seq_ids # a list. each element is the list of input history frames with length num_frames

        # # from TYVOS
        # self.prepare = ConvertCocoPolysToMask(return_masks)
        # self.argoverse = ArgoverseHD(json_file)
        # self.cat_ids = self.argoverse.getCatIds() # integer array of cat ids
        # self.vid_ids = self.argoverse.getVidIds()
        # self.vid_infos = []
        # for i in self.vid_ids:
        #     info = self.argoverse.loadVids([i])[0]
        #     info['filenames'] = info['file_names']
        #     self.vid_infos.append(info)
        # self.img_ids = []
        # for idx, vid_info in enumerate(self.vid_infos):
        #     for frame_id in range(len(vid_info['filenames'])):
        #         self.img_ids.append((idx, frame_id))

    def __len__(self):
        return len(self.target_im_ids)
    
    def load_images(self, idx):
        input_im_ids = self.input_seq_ids[idx]
        imgs = []
        for im_id in input_im_ids:
            image = self.api.imgs[im_id]
            img_path = os.path.join(str(self.data_dir),image['name'])
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
            imgs, target = self._transforms(imgs, target)

        return torch.cat(imgs,dim=0), target #return video ( [(C,H,W)]*num_frames -> (C*num_frame,H,W)), target


        # # from TYVOS
        # vid,  frame_id = self.img_ids[idx]
        # vid_id = self.vid_infos[vid]['id']
        # img = []
        # vid_len = len(self.vid_infos[vid]['file_names'])
        # inds = list(range(self.num_frames))
        # inds = [i%vid_len for i in inds][::-1]
        # # if random 
        # # random.shuffle(inds)
        # for j in range(self.num_frames):
        #     img_path = os.path.join(str(self.data_dir), self.vid_infos[vid]['file_names'][frame_id-inds[j]])
        #     img.append(Image.open(img_path).convert('RGB'))

        # ann_ids = self.argoverse.getAnnIds(vidIds=[vid_id])
        # target = self.argoverse.loadAnns(ann_ids)
        # target = {'image_id': idx, 'video_id': vid, 'frame_id': frame_id, 'annotations': target}
        # target = self.prepare(img[0], target, inds, self.num_frames)
        # if self._transforms is not None:
        #     img, target = self._transforms(img, target)
        # return torch.cat(imgs,dim=0), target #return video, target


# def convert_coco_poly_to_mask(segmentations, height, width):
#     masks = []
#     for polygons in segmentations:
#         if not polygons:
#             mask = torch.zeros((height,width), dtype=torch.uint8)
#         else:
#             rles = coco_mask.frPyObjects(polygons, height, width)
#             mask = coco_mask.decode(rles)
#             if len(mask.shape) < 3:
#                 mask = mask[..., None]
#             mask = torch.as_tensor(mask, dtype=torch.uint8)
#             mask = mask.any(dim=2)
#         masks.append(mask)
#     if masks:
#         masks = torch.stack(masks, dim=0)
#     else:
#         masks = torch.zeros((0, height, width), dtype=torch.uint8)
#     return masks


# class ConvertCocoPolysToMask(object):
#     def __init__(self, return_masks=False):
#         self.return_masks = return_masks

#     def __call__(self, image, target, inds, num_frames):
#         w, h = image.size
#         image_id = target["image_id"]
#         frame_id = target['frame_id']
#         image_id = torch.tensor([image_id])

#         anno = target["annotations"]
#         anno = [obj for obj in anno if 'iscrowd' not in obj or obj['iscrowd'] == 0]
#         boxes = []
#         classes = []
#         segmentations = []
#         area = []
#         iscrowd = []
#         valid = []
#         # add valid flag for bboxes
#         for i, ann in enumerate(anno):
#             for j in range(num_frames):
#                 bbox = ann['bboxes'][frame_id-inds[j]]
#                 areas = ann['areas'][frame_id-inds[j]]
#                 segm = ann['segmentations'][frame_id-inds[j]]
#                 clas = ann["category_id"]
#                 # for empty boxes
#                 if bbox is None:
#                     bbox = [0,0,0,0]
#                     areas = 0
#                     valid.append(0)
#                     clas = 0
#                 else:
#                     valid.append(1)
#                 crowd = ann["iscrowd"] if "iscrowd" in ann else 0
#                 boxes.append(bbox)
#                 area.append(areas)
#                 segmentations.append(segm)
#                 classes.append(clas)
#                 iscrowd.append(crowd)
#         # guard against no boxes via resizing
#         boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
#         boxes[:, 2:] += boxes[:, :2]
#         boxes[:, 0::2].clamp_(min=0, max=w)
#         boxes[:, 1::2].clamp_(min=0, max=h)
#         classes = torch.tensor(classes, dtype=torch.int64)
#         if self.return_masks:
#             masks = convert_coco_poly_to_mask(segmentations, h, w)
#         target = {}
#         target["boxes"] = boxes
#         target["labels"] = classes
#         if self.return_masks:
#             target["masks"] = masks
#         target["image_id"] = image_id

#         # for conversion to coco api
#         area = torch.tensor(area) 
#         iscrowd = torch.tensor(iscrowd)
#         target["valid"] = torch.tensor(valid)
#         target["area"] = area
#         target["iscrowd"] = iscrowd
#         target["orig_size"] = torch.as_tensor([int(h), int(w)])
#         target["size"] = torch.as_tensor([int(h), int(w)])
#         return  target


def make_coco_transforms(image_set):

    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]) # TODO:

    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768]

    if image_set == 'train':
        return T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomResize(scales, max_size=800),
            T.PhotometricDistort(),
            T.Compose([
                     T.RandomResize([400, 500, 600]),
                     T.RandomSizeCrop(384, 600),
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


def build(image_set, args):
    """
    image_set: 'train' or 'val'
    """
    root = Path(args.argoverse_path)
    assert root.exists(), f'provided dataset path {root} does not exist'
    # mode = 'instances'
    PATHS = {
            "train": (root / "train" / "JPEGImages", root / "annotations" / "instances_train_sub_argoformat.json"),
            "val": (root / "valid" / "JPEGImages", root / "annotations" / "instances_val_sub_argoformat.json")
    }
    if image_set in ['val','test']:
        print('WARNING: No annotations in {} set'.format(image_set))
    data_dir, json_file = PATHS[image_set]
    dataset = YTVISDataset(data_dir, json_file, transforms=make_coco_transforms(image_set), return_masks=args.masks, num_frames = args.num_frames, future=args.future)
    print(image_set, ' ',dataset.__len__(), ' measurements')
    return dataset

    