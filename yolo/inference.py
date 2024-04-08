import argparse
import os
import sys
from pathlib import Path

import cv2
import torch
import albumentations
import numpy as np
import torch.backends.cudnn as cudnn
from yolo.utils.augmentations import letterbox
from yolo.models.common import DetectMultiBackend
from yolo.utils.general import (non_max_suppression, print_args)
from yolo.utils.plots import Annotator, colors, save_one_box

def Get_Model(weights='./runs/train/exp/weights/best.pt', dnn=False):
    model = DetectMultiBackend(weights, device=torch.device('cuda:0'), dnn=dnn)
    return model

def resize_image(img_arr, bboxes, h, w):
    """
    :param img_arr: original image as a numpy array
    :param bboxes: bboxes as numpy array where each row is 'x_min', 'y_min', 'x_max', 'y_max', "class_id"
    :param h: resized height dimension of image
    :param w: resized weight dimension of image
    :return: dictionary containing {image:transformed, bboxes:['x_min', 'y_min', 'x_max', 'y_max', "class_id"]}
    """
    # create resize transform pipeline
    transform = albumentations.Compose(
        [albumentations.Resize(height=h, width=w, always_apply=True)],
        bbox_params=albumentations.BboxParams(format='pascal_voc'))
    transformed = transform(image=img_arr, bboxes=bboxes)
    return transformed

def adjust_coordinate(x0, y0, x1, y1, x, y):
    if x0 < 0:
        x0 = 0
    if y0 < 0:
        y0 = 0
    if x1 > x:
        x1 = x
    if y1 > y:
        y1 = y
    return x0, y0, x1, y1

def Inference(model, orginal_img, img, img_np, save_path, img_name, conf_thres = 0.2, iou_thres = 0.1, classes=None, agnostic_nms = False, max_det = 1):

    with torch.no_grad():
        pred = model(img, augment=False, visualize=False)
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        x_original, y_original = orginal_img.shape[1], orginal_img.shape[0]
        x_resized, y_resized = img_np.shape[0], img_np.shape[1]
        for i, det in enumerate(pred):
            for *xyxy, conf, cls in reversed(det):
                img_crop = orginal_img.copy()
                x0, y0, x1, y1 = xyxy[0].item(), xyxy[1].item(), xyxy[2].item(), xyxy[3].item()
                x0, y0, x1, y1 = adjust_coordinate(x0, y0, x1, y1, y_resized, x_resized)
                image = resize_image(img_np, [[x0, y0, x1, y1, 0]], y_original, x_original)
                x0, y0, x1, y1 = image['bboxes'][0][0], image['bboxes'][0][1], image['bboxes'][0][2], image['bboxes'][0][3]
                x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)
                cv2.rectangle(orginal_img, (x0, y0), (x1, y1), (0, 0, 255), 10)
                img_crop = img_crop[y0:y1, x0:x1]
                cv2.imwrite(save_path + img_name + '.png', img_crop)
        if len(pred) > 0:
            cv2.imwrite(save_path + img_name + '_annotated.png', orginal_img)


# def adjust_bbox(x0, y0, x1, y1):
#     height = y1 - y0
#     width = x1 - x0
#     y0 -= 0.5 * height  
#     y1 += 0.5 * height  
#     x0 -= 0.2 * width   
#     x1 += 0.2 * width   
#     return x0, y0, x1, y1

# def Inference_crop_box(model, orginal_img, img, img_np, crop_save_path, box_save_path, img_name, conf_thres = 0.2, iou_thres = 0.45, classes=None, agnostic_nms = False, max_det = 1):

#     with torch.no_grad():
#         pred = model(img, augment=False, visualize=False)
#         pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
#         x_original, y_original = orginal_img.shape[1], orginal_img.shape[0]
#         x_resized, y_resized = img_np.shape[0], img_np.shape[1]
#         for i, det in enumerate(pred):
#             for *xyxy, conf, cls in reversed(det):
#                 img_crop = orginal_img.copy()
#                 x0, y0, x1, y1 = xyxy[0].item(), xyxy[1].item(), xyxy[2].item(), xyxy[3].item()
#                 x0, y0, x1, y1 = adjust_coordinate(x0, y0, x1, y1, y_resized, x_resized)
#                 image = resize_image(img_np, [[x0, y0, x1, y1, 0]], y_original, x_original)
#                 x0, y0, x1, y1 = image['bboxes'][0][0], image['bboxes'][0][1], image['bboxes'][0][2], image['bboxes'][0][3]
#                 x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)
#                 cv2.rectangle(orginal_img, (x0, y0), (x1, y1), (0, 0, 255), 10)
#                 img_crop = img_crop[y0:y1, x0:x1]
#                 cv2.imwrite(crop_save_path + img_name, img_crop)
#         cv2.imwrite(box_save_path + img_name, orginal_img)


# def Inference_crop_tumor(model, orginal_img, img, img_np, crop_save_path, box_save_path, img_name, conf_thres = 0.2, iou_thres = 0.45, classes=None, agnostic_nms = False, max_det = 1):
#     with torch.no_grad():
#         pred = model(img, augment=False, visualize=False)
#         pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
#         x_original, y_original = orginal_img.shape[1], orginal_img.shape[0]
#         x_resized, y_resized = img_np.shape[1], img_np.shape[0]
#         for i, det in enumerate(pred):
        #     for *xyxy, conf, cls in reversed(det):
        #         img_crop = orginal_img.copy()
        #         x0, y0, x1, y1 = xyxy[0].item(), xyxy[1].item(), xyxy[2].item(), xyxy[3].item()
        #         if 2 * abs(x0-x1) >= abs(y0-y1):
        #             x0, y0, x1, y1 = adjust_coordinate(x0, y0, x1, y1, x_resized, y_resized)
        #             image = resize_image(img_np, [[x0, y0, x1, y1, 0]], y_original, x_original)
        #             x0, y0, x1, y1 = image['bboxes'][0][0], image['bboxes'][0][1], image['bboxes'][0][2], image['bboxes'][0][3]
        #             x0, y0, x1, y1 = adjust_bbox(x0, y0, x1, y1)
        #             x0, y0, x1, y1 = adjust_coordinate(x0, y0, x1, y1, x_original, y_original)
        #             x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)
        #             cv2.rectangle(orginal_img, (x0, y0), (x1, y1), (0, 0, 255), 10)
        #             img_crop = img_crop[y0:y1, x0:x1]
        #             cv2.imwrite(crop_save_path + img_name, img_crop)
        # cv2.imwrite(box_save_path + img_name, orginal_img)


# def Inference_crop_icon(model, img, conf_thres = 0.2, iou_thres = 0.45, classes=None, agnostic_nms = False, max_det = 1):
#     with torch.no_grad():
#         pred = model(img, augment=False, visualize=False)
#         pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
#         if len(pred[0]) > 0:
#             return True
#         else:
#             return False





