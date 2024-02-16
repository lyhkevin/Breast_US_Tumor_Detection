import os
from glob import glob
import pandas as pd
import shutil

base_path = 'D:/projects/breast_data_preprocessing/us_preprocessing/dataset/processed/NAC/tumor/'
NAC_1 = 'D:/private data/clinical data/云肿乳腺癌NAC队列20220712沈老师.xlsx'
NAC_2 = 'D:/private data/clinical data/云肿乳腺癌NAC队列沈老师第二批698例.xlsx'
NAC_3 = 'D:/private data/clinical data/乳腺癌NAC队列-沈定刚老师-第三批-20230117.xlsx'
save_path = 'D:/projects/breast_data_preprocessing/us_preprocessing/dataset/processed/NAC/Date/'


def get_date():
    clinical_1 = pd.read_excel(NAC_1)[['ID1', '新辅助治疗前原发灶病理收到时间']]
    clinical_2 = pd.read_excel(NAC_2)[['ID1', '新辅助治疗前原发灶病理收到时间']]
    clinical_3 = pd.read_excel(NAC_3)[['ID1', '新辅助治疗前原发灶病理收到时间']]
    clinical = pd.concat([clinical_1, clinical_2, clinical_3], axis=0)
    clinical = clinical.rename(columns={'ID1': 'id'})
    clinical = clinical.rename(columns={'新辅助治疗前原发灶病理收到时间': 'date'})
    clinical['date'] = pd.to_datetime(clinical['date'], errors='coerce')
    # Convert the datetime to a string format with length 8
    clinical['date'] = clinical['date'].dt.strftime('%Y%m%d')
    return clinical

def select_us_diagnosis(clinical):
    subject_paths = glob(base_path + '/*')
    for subject_path in subject_paths:
        dates = glob(subject_path + '/*/', recursive=True)
        for date in dates:
            num_images = len(glob(date + 'crop/*.png', recursive=True))
            if num_images >= 1:
                date_path = date.replace("\\", "/")
                id = date_path.split('/')[-3]
                date = date_path.split('/')[-2]
                if (clinical['id'] == int(id)).any():
                    diagnosis_date = clinical[clinical['id'] == int(id)]['date'].values[0]
                    if pd.isnull(diagnosis_date) == False:
                        if date <= diagnosis_date:
                            if len(glob(date_path + '/**/*.png', recursive=True)) > 0:
                                os.makedirs(save_path + id + '/diagnosis/', exist_ok=True)
                                shutil.copytree(date_path + '/crop/', save_path + id + '/diagnosis/' + date)
                        else:
                            if len(glob(date_path + '/**/*.png', recursive=True)) > 0:
                                os.makedirs(save_path + id + '/treatment/', exist_ok=True)
                                shutil.copytree(date_path + '/crop/', save_path + id + '/treatment/' + date)
                    else:
                        print('diagnose date is NaN')
                else:
                    print(id, 'do not have clinical')

if __name__ == '__main__':
    #empty_folders()
    clinical = get_date()
    select_us_diagnosis(clinical)
