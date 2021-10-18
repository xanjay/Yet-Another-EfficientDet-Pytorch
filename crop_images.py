# Author: Zylo117

"""
Simple Inference Script of EfficientDet-Pytorch
"""
import os
import glob
import tqdm
import sys
import time
import torch
from torch.backends import cudnn
from matplotlib import colors

from backbone import EfficientDetBackbone
import cv2
import numpy as np

from efficientdet.utils import BBoxTransform, ClipBoxes
from utils.utils import preprocess, invert_affine, postprocess, STANDARD_COLORS, standard_to_bgr, get_index_label, plot_one_box

INPUT_DIR = sys.argv[1] #or "/home/sanjay/freelance/Yet-Another-EfficientDet-Pytorch"
imgs_path = glob.glob(f'{INPUT_DIR}/*.jpg')

DEST_DIR = f'{INPUT_DIR}/cropped'
os.makedirs(f'{DEST_DIR}', exist_ok=True)

# print(sys.argv[1])


def write_crop(preds, imgs):
    for i in range(len(imgs)):

        imgs[i] = imgs[i].copy()
        big_boxes = []
        big_boxes_rect = []
        for j in range(len(preds[i]['rois'])):
            x1, y1, x2, y2 = preds[i]['rois'][j].astype(np.int)
            obj = obj_list[preds[i]['class_ids'][j]]
            score = float(preds[i]['scores'][j])
            if score > 0.50:
                big_boxes.append((y2-y1)*(x2-x1))
                big_boxes_rect.append([x1, y1, x2, y2])
                # plot_one_box(imgs[i], [x1, y1, x2, y2], label=obj,score=score,color=color_list[get_index_label(obj, obj_list)])


        try:
            large_box = big_boxes_rect[big_boxes.index(max(big_boxes))]
        except Exception:
            large_box = []
        
        if len(large_box):
            cropped_image = imgs[i][large_box[1]:large_box[3], large_box[0]:large_box[2], :]
        else:
            cropped_image = imgs[i]
        
        filename = img_path[i].split('/')[-1]
        dest_img_name = f'{DEST_DIR}/{filename}'
        # cv2.imwrite(f'{INPUT_DIR}/test/img_inferred_d{compound_coef}_this_repo_{i}.jpg', imgs[i])
        cv2.imwrite(dest_img_name, cropped_image)


compound_coef = 0
force_input_size = None  # set None to use default size


# replace this part with your project's anchor config
anchor_ratios = [(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)]
anchor_scales = [2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]

threshold = 0.2
iou_threshold = 0.2

use_cuda = True
use_float16 = False
cudnn.fastest = True
cudnn.benchmark = True

obj_list = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
            'fire hydrant', '', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
            'cow', 'elephant', 'bear', 'zebra', 'giraffe', '', 'backpack', 'umbrella', '', '', 'handbag', 'tie',
            'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
            'skateboard', 'surfboard', 'tennis racket', 'bottle', '', 'wine glass', 'cup', 'fork', 'knife', 'spoon',
            'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut',
            'cake', 'chair', 'couch', 'potted plant', 'bed', '', 'dining table', '', '', 'toilet', '', 'tv',
            'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
            'refrigerator', '', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
            'toothbrush']


color_list = standard_to_bgr(STANDARD_COLORS)
# tf bilinear interpolation is different from any other's, just make do
input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536, 1536]
input_size = input_sizes[compound_coef] if force_input_size is None else force_input_size


model = EfficientDetBackbone(compound_coef=compound_coef, num_classes=len(obj_list),
                             ratios=anchor_ratios, scales=anchor_scales)
model.load_state_dict(torch.load(f'weights/efficientdet-d{compound_coef}.pth', map_location='cpu'))
model.requires_grad_(False)
model.eval()

if use_cuda:
    model = model.cuda()
if use_float16:
    model = model.half()


# loop over images in batches
batch_size = 20
for img_idx in tqdm.tqdm(range(0, len(imgs_path), batch_size)):
    img_path = imgs_path[img_idx:img_idx+batch_size]

    ori_imgs, framed_imgs, framed_metas = preprocess(*img_path, max_size=input_size)

    if use_cuda:
        x = torch.stack([torch.from_numpy(fi).cuda() for fi in framed_imgs], 0)
    else:
        x = torch.stack([torch.from_numpy(fi) for fi in framed_imgs], 0)

    x = x.to(torch.float32 if not use_float16 else torch.float16).permute(0, 3, 1, 2)



    with torch.no_grad():
        features, regression, classification, anchors = model(x)

        regressBoxes = BBoxTransform()
        clipBoxes = ClipBoxes()

        out = postprocess(x,
                          anchors, regression, classification,
                          regressBoxes, clipBoxes,
                          threshold, iou_threshold)


    out = invert_affine(framed_metas, out)
    write_crop(out, ori_imgs)

print(f"images saved to '{DEST_DIR}'")

