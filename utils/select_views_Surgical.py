import os
import math
import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.models as models
from glob import glob
import torch.nn.functional as F
from PIL import Image
import shutil

def accumulate_decreasing(n):
    result = 0
    for i in range(n-1, 0, -1):
        result += i
    return result

origin_file = './dataset/tumor/'
save_file = './dataset/final/'

model = models.resnet18(pretrained = True)
model = torch.nn.Sequential(*(list(model.children())[:-1]))
model.eval()
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

patient = os.listdir(origin_file)

for i in range(len(patient)):
    patient_path = os.path.join(origin_file,patient[i])
    time = os.listdir(patient_path)
    time = time[0]
    time_path = os.path.join(patient_path, time)
    img = glob(time_path + '/crop/*.png', recursive=True)
    combine = accumulate_decreasing(len(img))
    idx_img1 = np.zeros(combine)
    idx_img2 = np.zeros(combine)
    mse = np.zeros(combine)
    x = 0
    if len(img) > 1:
        for i1 in range(0,len(img)-1):
            for i2 in range(i1+1,len(img)):
                image1 = Image.open(img[i1])
                image2 = Image.open(img[i2])

                image1_t = preprocess(image1).unsqueeze(0)
                image2_t = preprocess(image2).unsqueeze(0)

                feature1 = model(image1_t)
                feature2 = model(image2_t)

                mse[x] = F.mse_loss(feature1, feature2).cpu().item()
                idx_img1[x] = i1
                idx_img2[x] = i2

                x = x+1

        max_idx = np.argmax(mse, axis=0)
        
        destination_folder = save_file + '/' + patient[i] + '/'  # 替换为实际的目标文件夹路径

        if not os.path.exists(destination_folder):
            os.makedirs(destination_folder)
        shutil.copy(img[int(idx_img1[max_idx])], destination_folder)
        shutil.copy(img[int(idx_img2[max_idx])], destination_folder)

        print(patient[i], '  ', time, idx_img2[max_idx], idx_img1[max_idx])

    if len(img) == 1:
        destination_folder = save_file + '/' + patient[i] + '/' # 替换为实际的目标文件夹路径
        if not os.path.exists(destination_folder):
            os.makedirs(destination_folder)
        shutil.copy(img[0], destination_folder)
        print(patient[i], '  ', time, 0)








