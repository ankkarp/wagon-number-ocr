import gradio as gr
from os import walk
from model import NumberOcrModel
import pandas as pd
import csv
import cv2
import PIL
import numpy
import matplotlib
import os

pd_ans = pd.DataFrame(columns=['filemname', 'type', 'number', 'is_correct'])


def get_image(image_path):
    global pd_ans
    processed_image_path = None

    if image_path:

        res = model.predict(image_path)
        print(res)
        table_data = pd.DataFrame([res[0]])
        print(table_data)

        pd_ans = pd.DataFrame([list(res[0].values())], columns=['filemname', 'type', 'number', 'is_correct'])

        image_name = os.path.basename(os.path.normpath(image_path))

        all_dirs = os.listdir('./yolo_detections')
        max_length = len(max(all_dirs, key=len))
        data_dir = sorted([x for x in all_dirs if len(x) == max_length])[-1]
        dir_path = f'./yolo_detections/{data_dir}/'

        processed_image_path = dir_path + image_name
        print(processed_image_path)

    return processed_image_path, pd_ans


inputs = [
    gr.Image(type="filepath"),
]

outputs = [
    gr.Image(type='filepath'),
    gr.Dataframe(
            label="Результат обработки фото",
            row_count=3,
            max_rows=1,
            col_count=4,
    ),
]

demo = gr.Interface(get_image, inputs, outputs)
matplotlib.use('TkAgg')

if __name__ == '__main__':
    model = NumberOcrModel(
        detection_model='./models/custom_yolov8x.pt',  # path to detection_model
        rec_model='damo/cv_convnextTiny_ocr-recognition-general_damo',
        angle_rec_model='Aster'
    )

    demo.launch(share=True)