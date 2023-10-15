import argparse

from os import walk
import pandas as pd
import csv

from model2 import NumberOcrModel

parser = argparse.ArgumentParser()

parser.add_argument('-i', '--input', help='Input image folder', type=str, default='test_images/')
parser.add_argument('-o', '--output', help='Output csv-file', type=str, default='results.csv')
parser.add_argument('-d', '--detection-weights', help='Path to YOLO detection model weights', type=str, default='models/best(6).pt')
parser.add_argument('-r', '--rect-weights', help='Path to rectification model weights', type=str, default='moran.pth')


args = parser.parse_args()


if __name__ == "__main__":
    images = []
    results = []

    model = NumberOcrModel(
        detection_model=args.detection_weights,
        rec_model='damo/cv_convnextTiny_ocr-recognition-general_damo',
        angle_rec_model='SVTR-base',
    )

    for (dirpath, dirnames, filenames) in walk(args.input):
        images.extend(filenames)

    for img in images:
        res = model.predict(args.input + img)
        if res:
            results.append(res)

    with open(args.output, 'w', encoding='UTF8') as f:
        fields = ('filename', 'type', 'number', 'is_correct')
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator='\n')
        writer.writeheader()
        for res in results:
            writer.writerow(res[0])
    file = pd.read_csv(args.output)
    print(file)