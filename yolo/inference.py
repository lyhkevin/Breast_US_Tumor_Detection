import argparse
import os
import sys
from pathlib import Path

import cv2
import torch
import numpy as np
import torch.backends.cudnn as cudnn
from yolo.utils.augmentations import letterbox
from yolo.models.common import DetectMultiBackend
from yolo.utils.general import (non_max_suppression, print_args)
from yolo.utils.plots import Annotator, colors, save_one_box

def Get_Model(weights='./runs/train/exp/weights/best.pt', dnn=False):
    model = DetectMultiBackend(weights, device=torch.device('cuda:0'), dnn=dnn)
    return model

def Inference(model, img, img_np, crop_save_path, box_save_path, img_name, window_name, conf_thres = 0.2, iou_thres = 0.45, classes=None, agnostic_nms = False, max_det = 2):

    with torch.no_grad():
        pred = model(img, augment=False, visualize=False)
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        count = 1
        img_box = img_np.copy()
        num_boxes = pred[0].size()[0]
        for i, det in enumerate(pred):
            annotator = Annotator(img_box, line_width=5)
            for *xyxy, conf, cls in reversed(det):
                img_crop = img_np.copy()
                c = int(cls)  # integer class
                annotator.box_label(xyxy, label=f'box {conf:.2f}', color=colors(c, True))
                img_crop = save_one_box(xyxy, img_crop, BGR=True,save=False)
                if num_boxes > 1:
                    cv2.imwrite(crop_save_path + img_name + '_' + str(count) + '.png', img_crop)
                    count += 1
                else:
                    cv2.imwrite(crop_save_path + img_name + '.png', img_crop)
        cv2.imwrite(box_save_path + window_name + '.png', img_box)

if __name__ == "__main__":

    device = torch.device('cuda:0')
    model = Get_Model().to(device)
    input = cv2.imread('D:/projects/us_preprocessing/dataset/us_raw/5128/20210511162946/raw/0007018209_20210511_1_45.png')
    input_np = letterbox(input, 640, stride=32, auto=True)[0]
    input = input_np.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    input = np.ascontiguousarray(input)
    input = torch.from_numpy(input).unsqueeze(0).to(device)
    input = input.float()
    input /= 255
    Inference(model, input, input_np)
