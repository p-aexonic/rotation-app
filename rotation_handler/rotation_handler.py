import cv2
import numpy as np
import pytesseract
from PIL import Image, ExifTags, ImageFilter
from deskew import determine_skew
from pdf2image import convert_from_path
import img2pdf
import os
import logging
from typing import Generator
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
logging.basicConfig(level=logging.INFO)

class RotationHandler:
    def __init__(self, verify_rotation: bool = False):
        self.verify_rotation = verify_rotation

    def _fix_exif_rotation(self, image: Image.Image) -> Image.Image:
        try:
            if hasattr(image, '_getexif'):
                for orientation in ExifTags.TAGS.keys():
                    if ExifTags.TAGS[orientation] == 'Orientation':
                        break
                exif = image._getexif()
                if exif:
                    orientation_value = exif.get(orientation, None)
                    if orientation_value == 3:
                        image = image.rotate(180, expand=True)
                    elif orientation_value == 6:
                        image = image.rotate(270, expand=True)
                    elif orientation_value == 8:
                        image = image.rotate(90, expand=True)
        except Exception as e:
            logging.warning(f"EXIF rotation correction failed: {e}")
        return image

    def _crop_whitespace(self, img: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
        coords = cv2.findNonZero(thresh)
        x, y, w, h = cv2.boundingRect(coords)
        cropped = img[y:y+h, x:x+w]
        return cropped

    def _resize_image(self, img: np.ndarray, max_dim: int = 1000) -> np.ndarray:
        h, w = img.shape[:2]
        scale = max_dim / float(max(h, w))
        if scale < 1:
            img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        return img

    def _enhance_image(self, pil_image: Image.Image) -> Image.Image:
        pil_image = pil_image.convert("L").filter(ImageFilter.SHARPEN)
        return pil_image

    def _detect_orientation(self, image: Image.Image) -> int:
        try:
            # Resize image for better OCR accuracy
            resized = image.resize((image.width * 2, image.height * 2))
            osd = pytesseract.image_to_osd(resized)
            for line in osd.splitlines():
                if "Rotate:" in line:
                    return int(line.split(":")[-1].strip())
        except Exception as e:
            logging.warning(f"Orientation detection skipped: {e}")
        return 0  # fallback: assume correct

    def _rotate_image(self, img: np.ndarray, angle: int) -> np.ndarray:
        if angle == 0:
            return img
        (h, w) = img.shape[:2]
        center = (w // 2, h // 2)
        matrix = cv2.getRotationMatrix2D(center, -angle, 1.0)
        rotated = cv2.warpAffine(img, matrix, (w, h), flags=cv2.INTER_LINEAR)
        return rotated

    def _deskew_image(self, img: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        angle = determine_skew(gray)
        return self._rotate_image(img, angle)

    def _is_blank_image(self, img: np.ndarray) -> bool:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return np.std(gray) < 5

    def _process_image(self, pil_img: Image.Image) -> Image.Image:
        pil_img = self._fix_exif_rotation(pil_img)
        pil_img = self._enhance_image(pil_img)
        open_cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        open_cv_img = self._deskew_image(open_cv_img)
        open_cv_img = self._crop_whitespace(open_cv_img)
        open_cv_img = self._resize_image(open_cv_img)
        angle = self._detect_orientation(Image.fromarray(cv2.cvtColor(open_cv_img, cv2.COLOR_BGR2RGB)))
        open_cv_img = self._rotate_image(open_cv_img, angle)
        return Image.fromarray(cv2.cvtColor(open_cv_img, cv2.COLOR_BGR2RGB))

    def handle_image_rotation(self, input_path: str, output_path: str):
        try:
            img = Image.open(input_path)
            processed = self._process_image(img)
            processed.save(output_path)
            logging.info(f"Processed image saved to {output_path}")
        except Exception as e:
            logging.error(f"Failed to process image {input_path}: {e}")

    def _pdf_to_images_generator(self, pdf_path: str) -> Generator[Image.Image, None, None]:
        images = convert_from_path(pdf_path)
        for img in images:
            yield img

    def handle_pdf_rotation(self, input_pdf: str, output_pdf: str):
        try:
            processed_images = []
            for i, img in enumerate(self._pdf_to_images_generator(input_pdf)):
                processed = self._process_image(img)
                temp_file = f"temp_page_{i}.jpg"
                processed.save(temp_file, "JPEG")
                processed_images.append(temp_file)

            with open(output_pdf, "wb") as f:
                f.write(img2pdf.convert(processed_images))

            for temp_file in processed_images:
                if os.path.exists(temp_file):
                    os.remove(temp_file)

            logging.info(f"Processed PDF saved to {output_pdf}")

        except Exception as e:
            logging.error(f"Failed to process PDF {input_pdf}: {e}")