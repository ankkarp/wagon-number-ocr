import torchvision.transforms.functional as F
from mmocr.apis import MMOCRInferencer


if __name__ == "__main__":
    ocr = MMOCRInferencer(det=None, rec='Aster')
    stn = ocr.textrec_inferencer.model.data_preprocessor
    BATCH = ['EasyOCR/trainer/all_data/train/42030510.jpg']
    tf = ocr.textrec_inferencer.pipeline
    d = {'data_samples': [], 'inputs': []}
    for f in BATCH:
        for k, v in tf(f).items():
            d[k].append(v)
    pr = ocr.textrec_inferencer.model.data_preprocessor(d)
    out = ocr.textrec_inferencer.model.preprocessor(pr['inputs'])
    F.to_pil_image(out[0]).save('out.png')