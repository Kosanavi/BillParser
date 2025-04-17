from rapidocr import RapidOCR
from PIL import Image
import numpy as np
from PIL import ImageFilter

class OCRProcessor:
    def __init__(self):
        self.ocr = RapidOCR()

    def preprocess_image(self, image: Image.Image) -> Image.Image:
        # Convert to grayscale
        image = image.convert("L")
        # Apply Gaussian Blur
        image = image.filter(ImageFilter.GaussianBlur(1))
        # Binarization
        threshold = 128
        image = image.point(lambda p: 255 if p > threshold else 0)
        # Resize
        new_size = (int(image.width * 2), int(image.height * 2))
        return image.resize(new_size, Image.Resampling.LANCZOS)

    def extract_text(self, image: Image.Image) -> str:
        preprocessed_image = self.preprocess_image(image)  # Preprocess the image
        # Convert processed PIL image to NumPy array
        image_np = np.array(preprocessed_image)
        if image_np.shape[-1] == 4:
            image_np = image_np[:, :, :3]  # Drop alpha if it exists

        result = self.ocr(image_np)
        full_text = "\\n".join([line for line in result.txts])
        return full_text
