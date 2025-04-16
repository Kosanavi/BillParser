import numpy as np
from paddleocr import PaddleOCR
from PIL import Image

class OCRProcessor:
    def __init__(self):
        self.ocr = PaddleOCR(use_angle_cls=False, lang='en', det_model_dir='models/det', rec_model_dir='models/rec', use_gpu=False)

    def extract_text(self, image: Image.Image) -> str:
        result = self.ocr.ocr(np.array(image))
        full_text = "\n".join([line[1][0] for line in result[0]])
        return full_text