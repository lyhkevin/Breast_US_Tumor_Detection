import pydicom
from PIL import Image
from glob import glob
import re
import os
from tqdm import tqdm
import SimpleITK as sitk
import numpy as np

base_root_1 = '/public/home/liyh2022/projects/us_preprocessing/dataset/raw/US/'
base_root_2 = '/public/home/liyh2022/projects/us_preprocessing/dataset/raw/US_1-5619/'
save_root = '/public/home/liyh2022/projects/us_preprocessing/dataset/png/'

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

def mha_to_png(mha_path):
    image = sitk.ReadImage(mha_path)
    array = sitk.GetArrayFromImage(image)
    mask = np.where(array > 0, 255, 0).astype(np.uint8)
    mask_img = Image.fromarray(mask[0])
    return mask_img

def has_chinese_character(texts):
    texts = texts.split('\\')[-1]
    for text in texts:
        pattern = re.compile(r'[\u4e00-\u9fa5]')
        match = pattern.search(text)
        if match:
            return True
    return False

def has_label(texts):
    for text in texts:
        if text.endswith('.mha'):
            return True
    return False

def remove_single_digits(path):
    path = re.sub(r'\\+', '/', path)
    pattern = re.compile(r'/(\d)/')
    match = pattern.search(path)
    if match and len(match.group(1)) == 1:
        path = path[:match.start(1)] + path[match.end(1)+1:]
    return path

def extract_date_from_string(string):
    pattern = r'20\d{6}'
    match = re.search(pattern, string)
    if match:
        return match.group()
    else:
        return None

if __name__ == '__main__':
    subject_path = glob(base_root_1 + '*') + glob(base_root_2 + '*')
    os.makedirs(save_root, exist_ok=True)
    for path in tqdm(subject_path):
        path = path.replace("\\", "/")
        id = path.split('/')[-1]
        id = re.split(r'(\d+)', id)[1]
        subject_save_path = save_root + id + '/'
        img_paths = glob(os.path.join(path, '**/*.dcm'), recursive=True)
        for img_path in img_paths:
            img_path = img_path.replace("\\", "/")
            date = extract_date_from_string(img_path)
            if date != None:
                img = dcm_to_png(img_path)
                if img != None:
                    file_name = img_path.split('/')[-1].replace(".dcm", ".png")
                    save_path = subject_save_path + date + '/' + file_name
                    os.makedirs(subject_save_path + date, exist_ok=True)
                    img.save(save_path)
            else:
                print(img_path)