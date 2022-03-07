import wandb
import torch
import torch.nn.functional as F

def get_id2label_map(num_classes):
    if num_classes == 91+1:
        return coco_class_id_to_label
    elif num_classes == 9+1: #8+1
        return argoverse_class_id_to_label
    elif num_classes == 41+1: #40+1
        return ytvis_class_id_to_label
    else:
        return coco_class_id_to_label

def wandb_imgs_parser(samples,targets,outputs,idx=None):
    '''
    samples: nested_tensor
    targets:
    outputs: the model output {'pred_logits': ,'pred_boxes': , 'aux_outputs':}
    T = 1
    '''
    images, masks = samples.decompose()
    batch_size, num_query, num_classes = outputs['pred_logits'].shape
    id2label = get_id2label_map(num_classes)

    # target_sizes = torch.stack([t["size"] for t in targets], dim=0) # size before padding
    # results = postprocessors['bbox'](outputs, target_sizes) #[{'scores': s, 'labels': l, 'boxes': b}]*batch_size

    out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']
    prob = F.softmax(out_logits, -1)
    scores, labels = prob[..., :-1].max(-1)

    vis_data = []
    idx_range = idx if idx else range(batch_size)
    for i in idx_range:
        wandb_boxes = {}

        image = images[i]
        mask = masks[i]
        idx = torch.nonzero(~mask,as_tuple=True)
        oh,ow = idx[0].max()+1, idx[1].max()+1
        image = image[:,:oh,:ow]
        # result = results[i]
        target = targets[i]

        # prediction
        wandb_predictions = {"box_data": [],"class_labels": id2label}
        for n in range(num_query):
            box_data = {}
            box = out_bbox[i][n]
            box_data["position"] = {'middle':(box[0].item(),box[1].item()),'width':box[2].item(),'height':box[3].item()}
            class_id = labels[i][n].item()
            score = scores[i][n].item()
            box_data['class_id'] = class_id
            box_data['box_caption'] = "%s (%.3f)"%(id2label[int(class_id)],score)
            box_data['scores'] = {'score':score}
            wandb_predictions["box_data"].append(box_data)

        # ground_truth
        wandb_gt = {"box_data": [],"class_labels": id2label}
        for n in range(len(target['boxes'])):
            box_data = {}
            box = target["boxes"][n]
            box_data["position"] = {'middle':(box[0].item(),box[1].item()),'width':box[2].item(),'height':box[3].item()}
            class_id = target['labels'][n].item()
            box_data['class_id'] = class_id
            box_data['box_caption'] = id2label[int(class_id)]
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

ytvis_class_id_to_label = {
0:"UNKNOWN",
1:"person",
2:"giant_panda",
3:"lizard",
4:"parrot",
5:"skateboard",
6:"sedan",
7:"ape",
8:"dog",
9:"snake",
10: "monkey",
11: "hand",
12: "rabbit",
13: "duck",
14: "cat",
15: "cow",
16: "fish",
17: "train",
18: "horse",
19: "turtle",
20: "bear",
21: "motorbike",
22: "giraffe",
23: "leopard",
24: "fox",
25: "deer",
26: "owl",
27: "surfboard",
28: "airplane",
29: "truck",
30: "zebra",
31: "tiger",
32: "elephant",
33: "snowboard",
34: "boat",
35: "shark",
36: "mouse",
37: "frog",
38: "eagle",
39: "earless_seal",
40: "tennis_racket",
41: "EMPTY"
}

argoverse_class_id_to_label = { 0:'person',
                                1:'bicycle',
                                2:'car',
                                3:'motorcycle',
                                4:'bus',
                                5:'truck',
                                6:'traffic_light',
                                7:'stop_sign',
                                8:'empty'}