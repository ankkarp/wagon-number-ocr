import os
import re
from os import walk

import cv2
import torch
import torchvision.transforms.functional as F
from torch.autograd import Variable
from ultralytics import YOLO
from mmocr.apis import MMOCRInferencer
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
import torchvision.transforms.functional as F
from torchvision.transforms import ToPILImage
from collections import OrderedDict
from PIL import Image

from vaild_function import is_valid
from MORAN_v2.models.moran import MORAN
import MORAN_v2.tools.utils as utils
import MORAN_v2.tools.dataset as dataset


'''
    detection_mode: './models/custom_yolov8x.pt'
    rec_model: damo/cv_convnextTiny_ocr-recognition-general_damo
    angle_rec_model: Aster
'''

DETECTION_SAVE_PATH = './yolo_detections/results/crops/number/'
MODEL_RESULT_PATH = './model_result/results.csv'

BIN_TYPES = {
    'ADAPTIVE_THRESH_GAUSSIAN_C': cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    'ADAPTIVE_THRESH_MEAN_C': cv2.ADAPTIVE_THRESH_MEAN_C,
    'THRESH_OTSU': cv2.THRESH_OTSU
}


class NumberOcrModel:
    def __init__(self, detection_model, rec_model, angle_rec_model, moran_model):
        self.detection_model = YOLO(detection_model)
        self.rec_model = pipeline(Tasks.ocr_recognition, model=rec_model)
        self.angle_rec_model = MMOCRInferencer(rec=angle_rec_model)
        alphabet = '0:1:2:3:4:5:6:7:8:9:a:b:c:d:e:f:g:h:i:j:k:l:m:n:o:p:q:r:s:t:u:v:w:x:y:z:$'
        self.cuda_flag = False
        if torch.cuda.is_available():
            self.cuda_flag = True
            self.rectificator = MORAN(1, len(alphabet.split(':')), 256, 32, 100, BidirDecoder=True, CUDA=self.cuda_flag)
            self.rectificator = self.rectificator.cuda()
            state_dict = torch.load(moran_model)
        else:
            self.rectificator = MORAN(1, len(alphabet.split(':')), 256, 32, 100, BidirDecoder=True, inputDataType='torch.FloatTensor', CUDA=self.cuda_flag)
            state_dict = torch.load(moran_model, map_location='cpu')
        MORAN_state_dict_rename = OrderedDict()
        for k, v in state_dict.items():
            name = k.replace("module.", "") # remove `module.`
            MORAN_state_dict_rename[name] = v
        self.rectificator.load_state_dict(MORAN_state_dict_rename)
        for p in self.rectificator.parameters():
            p.requires_grad = False
        self.rectificator.eval()

        self.converter = utils.strLabelConverterForAttention(alphabet, ':')
        self.transformer = dataset.resizeNormalize((100, 32))

        self.detection_result = None
        self.img_path = None
        self.image_name = None
        self.bin_type = None
        self.rec_result = None

        self.prepare_model()

    def prepare_model(self):
        if torch.cuda.is_available():
            self.detection_model.to('cuda')

    def rectificate(self, crop_path):
        image = Image.open(crop_path).convert('L')
        image = self.transformer(image)

        if self.cuda_flag:
            image = image.cuda()
        image = image.view(1, *image.size())
        image = Variable(image)
        rect_crop = self.rectificator.rectify(image, test=True)
        # z = rect_crop[0] * torch.tensor(image.std())
        # z = z + torch.tensor(image.mean())
        # Image.fromarray(rect_crop).save(crop_path)
        F.to_pil_image(rect_crop[0].data.cpu().mul_(0.5).add_(0.5)).save(crop_path)


    def preprocess(self, image_path, image_name, bin_prep):
        detection_result = self.detection_model.predict(image_path, save=True, save_crop=True,
                                                        project='yolo_detections', name='results', verbose=False)
        all_dirs = os.listdir('./yolo_detections')
        max_length = len(max(all_dirs, key=len))
        data_dir = sorted([x for x in all_dirs if len(x) == max_length])[-1]
        dir_path = f'./yolo_detections/{data_dir}/crops/number/'

        crops = []
        for (dirpath, dirnames, filenames) in walk(dir_path):
            crops.extend(filenames)

        if len(crops) > 1:
            images = [cv2.imread(dir_path + img) for img in crops]
            crops_width = [img.shape[1] for img in images]
            crop_image_name = crops[crops_width.index(max(crops_width))]
        else:
            crop_image_name = image_name

        if detection_result and bin_prep:
            img = cv2.imread(dir_path + crop_image_name)
            blur_img = cv2.GaussianBlur(img, (1, 1), 0)
            bin_img = cv2.adaptiveThreshold(blur_img, 255, BIN_TYPES[bin_prep], cv2.THRESH_BINARY_INV, 29, -4)
            cv2.imwrite(dir_path + crop_image_name, bin_img)

        return detection_result, dir_path + crop_image_name

    def recognize(self, img_path, crop_image_path, detected_data):
        if not detected_data[0] or len(detected_data[0].cpu().numpy()) == 0:
            return [
                {
                    'filename': img_path,
                    'type': 0,
                    'number': 0,
                    'is_correct': 0,
                }
            ]
        self.rectificate(crop_image_path)

        result_1 = self.angle_rec_model(crop_image_path)
        result_2 = self.rec_model(crop_image_path)

        num_1 = re.sub(r'[^0-9]', '', result_1['predictions'][0]['rec_texts'][0])  # aster
        num_2 = re.sub(r'[^0-9]', '', result_2['text'][0])  # model scope

        if not num_2:
            num_sub = num_1
        elif num_1:
            if len(num_2) >= 8:
                num_sub = num_2[:8]
            if len(num_2) < 8 and len(num_1) >= len(num_2):
                num_sub = num_1[:8]
            if 8 > len(num_2) > len(num_1):
                num_sub = num_2
        else:
            num_sub = num_2
        print(num_sub)
        result = [{
            'filename': img_path,
            'type': int(num_sub != None),
            'number': int(num_sub) if num_sub != None and num_sub != '' else 0,
            'is_correct': is_valid(num_sub),
        }]

        return result

    def predict(self, img_path, bin_prep=None):
        image_name = os.path.basename(os.path.normpath(img_path))
        detected_data, crop_img_path = self.preprocess(img_path, image_name, bin_prep)
        return self.recognize(img_path, crop_img_path, detected_data)
