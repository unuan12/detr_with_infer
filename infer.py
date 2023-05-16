# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import random

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image, ImageDraw

import util.misc as utils
from models.backbone import build_backbone
from models.transformer import build_transformer
from models.detr import DETR, DETRsegm


def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--lr_backbone', default=0, type=float)
    parser.add_argument('--device', default='cpu',
                        help='device to use for training / testing')
    parser.add_argument('--image', default='./images/test2.jpg',
                        help='input image path')

    # Model parameters
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")
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
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=100, type=int,
                        help="Number of query slots")
    parser.add_argument('--pre_norm', action='store_true')

    # * Segmentation
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")
    
    # dataset parameters
    parser.add_argument('--dataset_file', default='coco')
    
    return parser

# standard PyTorch mean-std input image normalization
transform = T.Compose([
    # T.Resize(800),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# COCO classes
CLASSES = [
    'N/A', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A',
    'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
    'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack',
    'umbrella', 'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
    'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass',
    'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
    'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
    'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A',
    'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
    'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A',
    'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush'
]


def load_image(image):
    img = Image.open(image).resize((800,600))
    img_tensor = transform(img).unsqueeze(0)
    return img, img_tensor


def predict(model, img_tensor):
    with torch.no_grad():
        output = model(img_tensor)
    pred_logits = output['pred_logits'][0]
    pred_boxes = output['pred_boxes'][0]
    return pred_logits, pred_boxes


def plot_results(image,logits,boxes,fontsize):
    drw = ImageDraw.Draw(image)
    count = 0
    for logit, box in zip(logits, boxes):
        cls = logit.argmax()
        if cls >= len(CLASSES):  # if the class is larger than the length of CLASSES, we will just skip for now
            continue
        count += 1
        label = CLASSES[cls]
        box = box * torch.Tensor([800, 600, 800, 600])  # scale up the box to the original size
        x, y, w, h = box
        x0, x1 = x - w//2, x + w//2
        y0, y1 = y - h//2, y + h//2
        print('object {}: label:{},box:{}'.format(count,label,box))  # [x,y,w,h]
        drw.rectangle([x0,y0,x1,y1], outline='red',width=1)
        drw.text((x0,y0), label, 'red')
    print('{} objects found'.format(count))


def main(args):
    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))

    if args.frozen_weights is not None:
        assert args.masks, "Frozen training is meant for segmentation only"
    print(args)

    device = torch.device(args.device)

    num_classes = 20 if args.dataset_file != 'coco' else 91
    if args.dataset_file == "coco_panoptic":
        # for panoptic, we just add a num_classes that is large enough to hold
        # max_obj_id + 1, but the exact value doesn't really matter
        num_classes = 250
    device = torch.device(args.device)

    backbone = build_backbone(args)

    transformer = build_transformer(args)

    model = DETR(
        backbone,
        transformer,
        num_classes=num_classes,
        num_queries=args.num_queries,
        aux_loss=True
    )
    if args.masks:
        model = DETRsegm(model, freeze_detr=(args.frozen_weights is not None))
    model.to(device)

    if args.backbone == 'resnet50':
        model_path = './ckpt/detr-r50.pth'
    elif args.backbone == 'resnet101':
        model_path = './ckpt/detr-r101.pth'
    model.load_state_dict(state_dict=torch.load(model_path)['model'])
    img, img_tensor = load_image(args.image)
    pred_logits, pred_boxes = predict(model, img_tensor)
    img_cp = img.copy()
    plot_results(img_cp, pred_logits, pred_boxes, 15)
    img_cp.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser('DETR infer script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
