from rapidocr import RapidOCR
from PIL import Image
import numpy as np

class OCRProcessor:
    def __init__(self):
        self.ocr = RapidOCR()

    def extract_text(self, image: Image.Image) -> str:
        # Convert PIL image to NumPy array and then to BGR (RapidOCR expects BGR format)
        image = np.array(image)
        if image.shape[-1] == 4:
            image = image[:, :, :3]  # Drop alpha if exists

        result = self.ocr(image)
        full_text = "\n".join([line for line in result.txts])
        return full_text
