from glob import glob
import re
import os
import numpy as np
import torch
from tqdm import tqdm
from yolo.inference import *

if __name__ == '__main__':
    base_root = './dataset/png/'
    base_save_root = './dataset/box/'
    subject_paths = sorted(glob(base_root + '*'))
    device = torch.device('cuda:0')
    subject_paths.sort(key=lambda x: [int(c) if c.isdigit() else c for c in re.split(r'(\d+)', x)])
    model = Get_Model('./yolo/runs/train/crop_box/weights/best.pt').to(device)
    for subject_path in tqdm(subject_paths):
        dates = glob(subject_path + '/*')
        for date in dates:
            raw_paths = glob(os.path.join(date, '*.png'), recursive=True)
            for raw_path in raw_paths:
                raw_path = raw_path.replace("\\", "/")
                id = re.split(r'(\d+)', raw_path)[1]
                date = raw_path.split('/')[-2]
                crop_save_path = os.path.join(base_save_root, id, date, 'crop/')
                box_save_path = os.path.join(base_save_root, id, date, 'box/')
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
                Inference(model, orginal_img, input, input_np, crop_save_path, box_save_path, img_name)






