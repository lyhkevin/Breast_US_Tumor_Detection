# import pydicom
# from PIL import Image
# from glob import glob
# import re
# import os
# import cv2
# from tqdm import tqdm
# import numpy as np

# def get_number(folder_path):
#     longest_number = ''
#     numbers = re.findall(r'\d+', folder_path)
#     for number in numbers:
#         if len(number) > len(longest_number):
#             longest_number = number
#     return longest_number

# def dcm_to_png(dcm_path):
#     dcm = pydicom.dcmread(dcm_path)
#     if 'PlanarConfiguration' not in dcm:
#         dcm.PlanarConfiguration = 0
#     if 'PixelData' in dcm:
#         pixels = dcm.pixel_array
#         if len(pixels.shape) == 3:
#             pixels = cv2.cvtColor(pixels, cv2.COLOR_LAB2BGR)
#         img = Image.fromarray(pixels)
#         return img
#     else:
#         print('dcm_path, do not have pixel data')
#         return None
    
# path = '/public/home/liyh2022/projects/breast_data_preprocessing/benign/raw/benign_1489/US/*'
# save_root = '/public/home/liyh2022/projects/breast_data_preprocessing/benign/processed/US/png/'

# if __name__ == '__main__':
#     subject_path = glob(path + '*')
#     os.makedirs(save_root, exist_ok=True)
#     count = 0
#     for path in tqdm(subject_path):
#         path = path.replace("\\", "/")
#         id = get_number(path)
#         print(id)
#         subject_save_path = save_root + id + '/'
#         img_paths = glob(os.path.join(path, '*/*.dcm'), recursive=True)
#         for img_path in img_paths:
#             img_path = img_path.replace("\\", "/")
#             img = dcm_to_png(img_path)
#             if img != None:
#                 os.makedirs(subject_save_path, exist_ok=True)
#                 file_name = img_path.split('/')[-1].replace(".dcm", ".png")
#                 save_path = subject_save_path + '/' + file_name
#                 img.save(save_path)

import pydicom
from PIL import Image
from glob import glob
import re
import os
import cv2
from tqdm import tqdm
import numpy as np

def dcm_to_png(dcm_path):
    dcm = pydicom.dcmread(dcm_path)
    if 'PlanarConfiguration' not in dcm:
        dcm.PlanarConfiguration = 0
    if 'PixelData' in dcm:
        pixels = dcm.pixel_array
        img = Image.fromarray(pixels)
        return img
    else:
        print('dcm_path, do not have pixel data')
        return None
    
path = '/public/home/liyh2022/projects/breast_data_preprocessing/benign/raw/inflammation_405/US/*'
save_root = '/public/home/liyh2022/projects/breast_data_preprocessing/benign/processed/US/png/'

if __name__ == '__main__':
    subject_path = glob(path + '*')
    os.makedirs(save_root, exist_ok=True)
    for path in tqdm(subject_path):
        path = path.replace("\\", "/")
        id = path.split('/')[-1]
        id = re.split(r'(\d+)', id)[1]
        subject_save_path = save_root + id + '/'
        img_paths = glob(os.path.join(path, '*/*.dcm'))
        for img_path in img_paths:
            img_path = img_path.replace("\\", "/")
            img = dcm_to_png(img_path)
            if img != None:
                os.makedirs(subject_save_path, exist_ok=True)
                file_name = img_path.split('/')[-1].replace(".dcm", ".png")
                save_path = subject_save_path +  '/' + file_name
                img.save(save_path)