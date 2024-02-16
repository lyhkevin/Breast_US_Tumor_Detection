import numpy as np
import pandas as pd
import re
from glob import glob
import os
from tqdm import tqdm
import shutil

clinical_1 = 'D:/projects/breast_data_preprocessing/benign/raw/乳腺良性2021-1200-已收集完.xlsx'
clinical_2 = 'D:/projects/breast_data_preprocessing/benign/raw/乳腺炎症病变-已收集完-20240119.xlsx'
clinical_1 = pd.read_excel(clinical_1)
clinical_2 = pd.read_excel(clinical_2)
subject_root = 'D:/projects/breast_data_preprocessing/benign/processed/US/tumor/'
save_root = 'D:/projects/breast_data_preprocessing/benign/processed/US/final/'
os.makedirs(save_root, exist_ok=True)

subject_paths = glob(subject_root + '*')
clinical_1['住院号'] = clinical_1['住院号'].astype(str)
clinical_2['住院号'] = clinical_2['住院号'].astype(str)
num_without_clinical = 0
num_with_clinical = 0
for subject_path in tqdm(subject_paths):
    id = subject_path.split('\\')[-1]
    id = id.lstrip("0")
    rows_clinical_1 = clinical_1[clinical_1['住院号'] == id]
    rows_clinical_2 = clinical_2[clinical_2['住院号'] == id]
    if rows_clinical_1.shape[0] == 0 and rows_clinical_2.shape[0] == 0:
        num_without_clinical += 1
        continue
    num_with_clinical += 1
    if rows_clinical_1.shape[0] > 0:
        new_id = rows_clinical_1['编号'].values[0]
    else:
        description = rows_clinical_2['医生主要诊断描述'].values[0]
        new_id = rows_clinical_2['编号'].values[0]
    #move to final folder
    print(subject_path, new_id)
    if not os.path.exists(os.path.join(save_root, str(new_id))):
        shutil.copytree(subject_path, os.path.join(save_root, str(new_id)))

print('num_without_clinical', num_without_clinical)
print('num_with_clinical', num_with_clinical)
