import pydicom
from PIL import Image
from glob import glob
from yolo.inference import *

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

if __name__ == '__main__':
    dcm_paths = glob('./data/*.dcm')
    save_path = './cropped/'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    crop_box_model = Get_Model('./yolo/yolo_weight/crop_box.pt').to(device)
    crop_tumor_model = Get_Model('./yolo/yolo_weight/crop_tumor.pt').to(device)
    for dcm_path in dcm_paths:
        file_name = os.path.basename(dcm_path)
        file_name = file_name.split('.')[0]
        os.makedirs(save_path + file_name, exist_ok=True)
        img = dcm_to_png(dcm_path)
        if img != None:
            img.save(save_path + file_name +  '/original.png')

            #crop window from the interface
            subject_path = save_path + file_name + '/'
            original_img = cv2.imread(subject_path + '/original.png')
            input_np = letterbox(original_img, 640, stride=32, auto=True)[0]
            input = input_np.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
            input = np.ascontiguousarray(input)
            input = torch.from_numpy(input).unsqueeze(0).to(device)
            input = input.float()
            input /= 255
            Inference(model = crop_box_model, orginal_img = original_img, img = input, img_np = input_np, save_path = subject_path, conf_thres = 0.1, img_name = 'image', max_det = 1)

            # crop tumor
            original_img = cv2.imread(subject_path + '/image.png')
            input_np = letterbox(original_img, 640, stride=32, auto=True)[0]
            input = input_np.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
            input = np.ascontiguousarray(input)
            input = torch.from_numpy(input).unsqueeze(0).to(device)
            input = input.float()
            input /= 255
            Inference(model = crop_tumor_model, orginal_img = original_img, img = input, img_np = input_np, save_path = subject_path, conf_thres = 0.05, img_name = 'tumor', max_det = 1)



