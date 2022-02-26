# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Train and eval functions used in main.py
"""
import wandb
import math
import os
import sys
from typing import Iterable

import torch

import util.misc as utils
from datasets.coco_eval import CocoEvaluator
from datasets.panoptic_eval import PanopticEvaluator


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module, postprocessors,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0):
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    i=0
    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = model(samples)
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        #log every batch
        if 'LOCAL_RANK' not in os.environ or int(os.environ['LOCAL_RANK']) == 0:
            wandb_img = wandb_imgs_parser(samples,targets,outputs,postprocessors,idx=[0])[0]
            wandb.log({"train_vis":wandb_img},commit=False)
            wandb.log({"train/loss_value":loss_value}, commit=False)
            wandb.log({f"train/{k}_unweighted":loss_dict_reduced_unscaled[k] for k in loss_dict_reduced_unscaled}, commit=False)
            wandb.log({f"train/{k}_weighted":loss_dict_reduced_scaled[k] for k in loss_dict_reduced_scaled}, commit=False)
            wandb.log({'epoch':epoch},step=len(data_loader)*epoch+i)
            i+=1
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(model, criterion, postprocessors, data_loader, base_ds, device, output_dir):
    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'

    iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())
    coco_evaluator = CocoEvaluator(base_ds, iou_types)
    # coco_evaluator.coco_eval[iou_types[0]].params.iouThrs = [0, 0.1, 0.5, 0.75]

    panoptic_evaluator = None
    if 'panoptic' in postprocessors.keys():
        panoptic_evaluator = PanopticEvaluator(
            data_loader.dataset.ann_file,
            data_loader.dataset.ann_folder,
            output_dir=os.path.join(output_dir, "panoptic_eval"),
        )

    for samples, targets in metric_logger.log_every(data_loader, 10, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = model(samples)
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()),
                             **loss_dict_reduced_scaled,
                             **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessors['bbox'](outputs, orig_target_sizes)
        if 'segm' in postprocessors.keys():
            target_sizes = torch.stack([t["size"] for t in targets], dim=0)
            results = postprocessors['segm'](results, outputs, orig_target_sizes, target_sizes)
        res = {target['image_id'].item(): output for target, output in zip(targets, results)}
        if coco_evaluator is not None:
            coco_evaluator.update(res)

        if panoptic_evaluator is not None:
            res_pano = postprocessors["panoptic"](outputs, target_sizes, orig_target_sizes)
            for i, target in enumerate(targets):
                image_id = target["image_id"].item()
                file_name = f"{image_id:012d}.png"
                res_pano[i]["image_id"] = image_id
                res_pano[i]["file_name"] = file_name

            panoptic_evaluator.update(res_pano)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()
    if panoptic_evaluator is not None:
        panoptic_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()
    panoptic_res = None
    if panoptic_evaluator is not None:
        panoptic_res = panoptic_evaluator.summarize()
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    if coco_evaluator is not None:
        if 'bbox' in postprocessors.keys():
            bbox_stats = coco_evaluator.coco_eval['bbox'].stats.tolist()
            stats['coco_eval_bbox'] = bbox_stats
            if 'LOCAL_RANK' not in os.environ or int(os.environ['LOCAL_RANK']) == 0:
                wandb_img = wandb_imgs_parser(samples,targets,outputs,postprocessors,idx=[0])[0]
                wandb.log({'val_vis':wandb_img},commit=False)
                wandb.log({'val/map':bbox_stats[0],
                            'val/map_50':bbox_stats[1],
                            'val/map_75':bbox_stats[2],
                            'val/map_small':bbox_stats[3],
                            'val/map_medium':bbox_stats[4],
                            'val/map_large':bbox_stats[5],
                            'val/mar_1':bbox_stats[6],
                            'val/mar_10':bbox_stats[7],
                            'val/mar_100':bbox_stats[8],
                            'val/mar_small':bbox_stats[9],
                            'val/mar_medium':bbox_stats[10],
                            'val/mar_large':bbox_stats[11]},commit=False)
        if 'segm' in postprocessors.keys():
            stats['coco_eval_masks'] = coco_evaluator.coco_eval['segm'].stats.tolist()
    if panoptic_res is not None:
        stats['PQ_all'] = panoptic_res["All"]
        stats['PQ_th'] = panoptic_res["Things"]
        stats['PQ_st'] = panoptic_res["Stuff"]
    return stats, coco_evaluator

def wandb_imgs_parser(samples,targets,outputs,postprocessors,idx=None):
    '''
    samples: nested_tensor
    targets:
    outputs: {'pred_logits': ,'pred_boxes': , 'aux_outputs':}
    T = 1
    '''
    images, masks = samples.decompose()
    batch_size, num_query, num_classes = outputs['pred_logits'].shape

    target_sizes = torch.stack([t["size"] for t in targets], dim=0)
    results = postprocessors['bbox'](outputs, target_sizes) #[{'scores': s, 'labels': l, 'boxes': b}]*batch_size

    vis_data = []
    idx_range = idx if idx else range(batch_size)
    for i in idx_range:
        wandb_boxes = {}

        image = images[i]
        mask = masks[i]
        idx = torch.nonzero(~mask,as_tuple=True)
        oh,ow = idx[0].max()+1, idx[1].max()+1
        image = image[:,:oh,:ow]
        result = results[i]
        target = targets[i]

        # prediction
        wandb_predictions = {"box_data": [],"class_labels": coco_class_id_to_label}
        for n in range(num_query):
            box_data = {}
            box = result["boxes"][n]
            box_data["position"] = {'minX':box[0].item(),'maxX':box[1].item(),'minY':box[2].item(),'maxY':box[3].item()}
            class_id = result['labels'][n].item()
            score = result['scores'][n].item()
            box_data['class_id'] = class_id
            box_data['box_caption'] = "%s (%.3f)"%(coco_class_id_to_label[int(class_id)],score)
            box_data['scores'] = {'score':score}
            box_data['domain'] = 'pixel'
            wandb_predictions["box_data"].append(box_data)

        # ground_truth
        wandb_gt = {"box_data": [],"class_labels": coco_class_id_to_label}
        for n in range(len(target['boxes'])):
            box_data = {}
            box = target["boxes"][n]
            box_data["position"] = {'middle':(box[0].item(),box[1].item()),'width':box[2].item(),'height':box[3].item()}
            class_id = target['labels'][n].item()
            box_data['class_id'] = class_id
            box_data['box_caption'] = coco_class_id_to_label[int(class_id)]
            wandb_gt["box_data"].append(box_data)
        wandb_boxes = {'predictions':wandb_predictions, 'ground_truth':wandb_gt}
        wandb_img = wandb.Image(image, boxes=wandb_boxes)
        vis_data.append(wandb_img)

    return vis_data

coco_class_id_to_label={
 1: "person",
 2: "bicycle",
 3: "car",
 4: "motorcycle",
 5: "airplane",
 6: "bus",
 7: "train",
 8: "truck",
 9: "boat",
10: "traffic light",
11: "fire hydrant",
12: "unknown",
13: "stop sign",
14: "parking meter",
15: "bench",
16: "bird",
17: "cat",
18: "dog",
19: "horse",
20: "sheep",
21: "cow",
22: "elephant",
23: "bear",
24: "zebra",
25: "giraffe",
26: "unknown",
27: "backpack",
28: "umbrella",
29: "unknown",
30: "unknown",
31: "handbag",
32: "tie",
33: "suitcase",
34: "frisbee",
35: "skis",
36: "snowboard",
37: "sports ball",
38: "kite",
39: "baseball bat",
40: "baseball glove",
41: "skateboard",
42: "surfboard",
43: "tennis racket",
44: "bottle",
45: "unknown",
46: "wine glass",
47: "cup",
48: "fork",
49: "knife",
50: "spoon",
51: "bowl",
52: "banana",
53: "apple",
54: "sandwich",
55: "orange",
56: "broccoli",
57: "carrot",
58: "hot dog",
59: "pizza",
60: "donut",
61: "cake",
62: "chair",
63: "couch",
64: "potted plant",
65: "bed",
66: "unknown",
67: "dining table",
68: "unknown",
69: "unknown",
70: "toilet",
71: "unknown",
72: "tv",
73: "laptop",
74: "mouse",
75: "remote",
76: "keyboard",
77: "cell phone",
78: "microwave",
79: "oven",
80: "toaster",
81: "sink",
82: "refrigerator",
83: "unknown",
84: "book",
85: "clock",
86: "vase",
87: "scissors",
88: "teddy bear",
89: "hair drier",
90: "toothbrush",
91: "empty"
}