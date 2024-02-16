from glob import glob
import re
import os
import numpy as np
import torch
from tqdm import tqdm
from yolo.inference import *

if __name__ == '__main__':
    base_root = '/public/home/liyh2022/projects/breast_data_preprocessing/benign/processed/US/png/'
    base_save_root = '/public/home/liyh2022/projects/breast_data_preprocessing/benign/processed/US/box/'
    subject_paths = sorted(glob(base_root + '*'))
    device = torch.device('cuda:0')
    subject_paths.sort(key=lambda x: [int(c) if c.isdigit() else c for c in re.split(r'(\d+)', x)])
    model = Get_Model('./yolo/runs/train/crop_box/weights/best.pt').to(device)
    for subject_path in tqdm(subject_paths):
        id = subject_path.split('/')[-1]
        print(subject_path, id)
        raw_paths = glob(subject_path + '/*.png')
        for raw_path in raw_paths:
            raw_path = raw_path.replace("\\", "/")
            crop_save_path = os.path.join(base_save_root, id, 'crop/')
            box_save_path = os.path.join(base_save_root, id, 'box/')
            os.makedirs(crop_save_path, exist_ok=True)
            os.makedirs(box_save_path, exist_ok=True)
            
            img_name = raw_path.split('/')[-1]
            orginal_img = cv2.imread(raw_path)
            input_np = letterbox(orginal_img, 640, stride=32, auto=True)[0]
            input = input_np.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
            input = np.ascontiguousarray(input)
            input = torch.from_numpy(input).unsqueeze(0).to(device)
            input = input.float()
            input /= 255
            Inference_crop_box(model, orginal_img, input, input_np, crop_save_path, box_save_path, img_name, conf_thres = 0.2, iou_thres = 0.45)






