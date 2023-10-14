from ultralytics import YOLO
from mmocr.apis import MMOCRInferencer
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
import torch
import cv2
import os
import re
from vaild_function import is_valid

'''
    detection_mode: PATH TO DETECTION MODEL
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
    def __init__(self, detection_model, rec_model, angle_rec_model):
        self.detection_model = YOLO(detection_model)
        self.rec_model = pipeline(Tasks.ocr_recognition, model=rec_model)
        self.angle_rec_model = MMOCRInferencer(rec=angle_rec_model)

        self.detection_result = None
        self.img_path = None
        self.image_name = None
        self.bin_type = None
        self.rec_result = None

        self.prepare_model()

    def prepare_model(self):
        if torch.cuda.is_available():
            self.detection_model.to('cuda')

    def preprocess(self, image_path, image_name, bin_prep):
        detection_result = self.detection_model.predict(image_path, save=True, save_crop=True,
                                                        project='yolo_detections', name='results', verbose=False)

        if detection_result and bin_prep:
            img = cv2.imread(DETECTION_SAVE_PATH + image_name)
            blur_img = cv2.GaussianBlur(img, (1, 1), 0)
            bin_img = cv2.adaptiveThreshold(blur_img, 255, BIN_TYPES[bin_prep], cv2.THRESH_BINARY_INV, 29, -4)
            cv2.imwrite(DETECTION_SAVE_PATH + image_name, bin_img)
        return detection_result

    def recognize(self, image_name, detected_data):
        if not detected_data[0] or len(detected_data[0].cpu().numpy()) == 0:
            return [
                {
                    'filename': image_name,
                    'type': 0,
                    'number': 0,
                    'is_correct': 0,
                }]

        all_dirs = os.listdir('./yolo_detections')
        max_length = len(max(all_dirs, key=len))
        data_dir = sorted([x for x in all_dirs if len(x) == max_length])[-1]

        crop_img_path = f'./yolo_detections/{data_dir}/crops/number/' + image_name

        result_1 = self.angle_rec_model(crop_img_path)
        result_2 = self.rec_model(crop_img_path)

        num_1 = re.sub(r'[^0-9]', '', result_1['predictions'][0]['rec_texts'][0])  # aster
        num_2 = re.sub(r'[^0-9]', '', result_2['text'][0])  # model scope

        if len(num_2) == 8:
            num_sub = num_2
        elif len(num_1) == 8:
            num_sub = num_1
        elif len(num_2) > 8:
            num_sub = num_2[:8]
        elif len(num_1) > 8:
            num_sub = num_1[:8]
        elif not num_2:
            num_sub = num_1
        elif not num_1:
            num_sub = num_2
        elif len(num_2) < 8 and len(num_1) == 8:
            num_sub = num_1
        else:
            num_sub = num_2

        result = [{
            'filename': image_name,
            'type': (0, 1)[len(num_sub) > 0],
            'number': num_sub,
            'is_correct': is_valid(num_sub),
        }]

        return result

    def predict(self, img_path, bin_prep=None):
        image_name = os.path.basename(os.path.normpath(img_path))
        detected_data = self.preprocess(img_path, image_name, bin_prep)
        return self.recognize(image_name, detected_data)
