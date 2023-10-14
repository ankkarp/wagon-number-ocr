import os
from os import walk
from model import NumberOcrModel
import pandas as pd
import csv

MODEL_RESULT_PATH = 'results.csv'  # save path for csv with results
# DATA_PATH = './test_images/'  # path to test images
DATA_PATH = './small_test/'

images = []
results = []

if not os.path.isdir('./models'):
    os.mkdir('./models')
if not os.path.isdir('./models'):
    os.mkdir('./test_images')


def to_csv(results):
    with open(MODEL_RESULT_PATH, 'w', encoding='UTF8') as f:
        fields = ('filename', 'type', 'number', 'is_correct')
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator='\n')
        writer.writeheader()
        for res in results:
            writer.writerow(res[0])


model = NumberOcrModel(
    detection_model='./models/custom_yolov8x.pt',  # path to detection_model
    rec_model='damo/cv_convnextTiny_ocr-recognition-general_damo',
    angle_rec_model='Aster'
)

for (dirpath, dirnames, filenames) in walk(DATA_PATH):
    images.extend(filenames)

for img in images:
    res = model.predict(DATA_PATH + img)
    if res:
        results.append(res)

to_csv(results)
file = pd.read_csv(MODEL_RESULT_PATH)
print(file)
