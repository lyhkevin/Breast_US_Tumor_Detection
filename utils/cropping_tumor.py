from glob import glob
import re
import os
import numpy as np
import torch
from tqdm import tqdm
import shutil
from yolo.inference import *

if __name__ == '__main__':
    base_root = './dataset/box/'
    base_save_root = './dataset/tumor/'
    subject_path = sorted(glob(base_root + '/*'))
    device = torch.device('cuda:0')
    subject_path.sort(key=lambda x: [int(c) if c.isdigit() else c for c in re.split(r'(\d+)', x)])
    model = Get_Model('./yolo/runs/train/crop_tumor/weights/best.pt').to(device)
    for path in tqdm(subject_path):
        dates = glob(path + '/*')
        for date in dates:
            raw_paths = glob(os.path.join(date, 'crop/*.png'), recursive=True)
            for raw_path in raw_paths:
                raw_path = raw_path.replace("\\", "/")
                id = re.split(r'(\d+)', raw_path)[1]
                date = raw_path.split('/')[-3]
                img_name = raw_path.split('/')[-1]
                crop_save_path = os.path.join(base_save_root, id, date, 'crop/')
                box_save_path = os.path.join(base_save_root, id, date, 'box/')
                os.makedirs(crop_save_path, exist_ok=True)
                os.makedirs(box_save_path, exist_ok=True)

                orginal_img = cv2.imread(raw_path)
                input_np = letterbox(orginal_img, 640, stride=32, auto=True)[0]
                input = input_np.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
                input = np.ascontiguousarray(input)
                input = torch.from_numpy(input).unsqueeze(0).to(device)
                input = input.float()
                input /= 255
                Inference(model, orginal_img, input, input_np, crop_save_path, box_save_path, img_name, conf_thres = 0.15, max_det = 1)
            
            png_files = glob(box_save_path + '*.png', recursive=True)
            if len(png_files) == 0:
                shutil.rmtree(os.path.join(box_save_path))
                shutil.rmtree(os.path.join(crop_save_path))










