import os
import re
from os import walk

import cv2
import torch
from ultralytics import YOLO
from mmocr.apis import MMOCRInferencer
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
import torchvision.transforms.functional as F

from vaild_function import is_valid


DETECTION_SAVE_PATH = './yolo_detections/results/crops/number/'
MODEL_RESULT_PATH = './model_result/results.csv'

BIN_TYPES = {
    'ADAPTIVE_THRESH_GAUSSIAN_C': cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    'ADAPTIVE_THRESH_MEAN_C': cv2.ADAPTIVE_THRESH_MEAN_C,
    'THRESH_OTSU': cv2.THRESH_OTSU
}


class NumberOcrModel:
    def __init__(self, detection_model, rec_model, angle_rec_model):
        '''
            detection_model: './models/custom_yolov8x.pt'
            rec_model: damo/cv_convnextTiny_ocr-recognition-general_damo
            angle_rec_model: Aster
        '''
        self.detection_model = YOLO(detection_model)
        self.rec_model = pipeline(Tasks.ocr_recognition, model=rec_model)
        self.angle_rec_model = MMOCRInferencer(rec=angle_rec_model)
        self.tfms = self.angle_rec_model.textrec_inferencer.pipeline

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
        d = {k: [v] for k, v in self.tfms(crop_path).items()}
        pr = self.angle_rec_model.textrec_inferencer.model.data_preprocessor(d)
        rect_crop = self.angle_rec_model.textrec_inferencer.model.preprocessor(pr['inputs'])
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

    def recognize(self, image_name, crop_image_path, detected_data):
        if not detected_data[0] or len(detected_data[0].cpu().numpy()) == 0:
            return [
                {
                    'filename': image_name,
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

        result = [{
            'filename': image_name,
            'type': int(num_sub != None),
            'number': (0, num_sub)[num_sub != None and num_sub != ''],
            'is_correct': is_valid(num_sub),
        }]

        return result

    def predict(self, img_path, bin_prep=None):
        image_name = os.path.basename(os.path.normpath(img_path))
        detected_data, crop_img_path = self.preprocess(img_path, image_name, bin_prep)
        return self.recognize(image_name, crop_img_path, detected_data)
