from rapidocr import RapidOCR
from PIL import Image
import numpy as np
from PIL import ImageFilter

class OCRProcessor:
    def __init__(self):
        self.ocr = RapidOCR()

    def extract_text(self, image: Image.Image) -> str:
        # Convert processed PIL image to NumPy array
        image_np = np.array(image)
        if image_np.shape[-1] == 4:
            image_np = image_np[:, :, :3]  # Drop alpha if it exists

        result = self.ocr(image_np)
        full_text = "\\n".join([line for line in result.txts])
        return full_text
