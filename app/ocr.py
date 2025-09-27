import cv2
import numpy as np
import pytesseract
from pytesseract import Output
from PIL import Image, ImageEnhance
from typing import List, Dict, Any
import os
import re

# ---------------- ENV ----------------
TESSERACT_CMD = os.getenv("TESSERACT_CMD")
if TESSERACT_CMD:
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD

# ---------------- OCR PREPROCESSING ----------------
def preprocess_medical_report_resize_sharpen(img_path, output_path=None, scale_factor=2.0):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Image not found: {img_path}")

    new_size = (int(img.shape[1] * scale_factor), int(img.shape[0] * scale_factor))
    resized = cv2.resize(img, new_size, interpolation=cv2.INTER_LINEAR)
    blurred = cv2.medianBlur(resized, 3)
    sharpened = cv2.addWeighted(resized, 2.4, blurred, -1.6, 0)
    _, thresh = cv2.threshold(sharpened, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)
    processed_img = cv2.bitwise_not(closed)

    # Convert to PIL Image for enhancement
    pil_img = Image.fromarray(processed_img)
    enhancer = ImageEnhance.Sharpness(pil_img)
    sharpened_pil = enhancer.enhance(9.0)

    # Convert back to numpy for saving
    sharpened_image = np.array(sharpened_pil)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if output_path:
        cv2.imwrite(output_path, sharpened_image)

    return sharpened_image

# ---------------- OCR EXTRACTION ----------------
def ocr_txt_extraction(path: str) -> str:
    img = Image.open(path)
    txt = pytesseract.image_to_string(img, lang='eng')
    return txt
def extract_medical_test_data(txt: str) -> List:
    # Split on any sequence of one or more blank lines
    lines = re.split(r'\n\s*\n', txt)
    # Remove any leading/trailing whitespace from each chunk
    lines = [chunk.strip() for chunk in lines if chunk.strip()]
    res = []
    for idx, line in enumerate(lines):
        if any(keyword in line.lower() for keyword in ('test', 'result', 'unit', 'value')):
            candidate_lines = lines[idx+1:]
            
            # Split into sub-lines
            sub_lines = []
            for chunk in candidate_lines:
                sub_lines.extend(chunk.split('\n'))
            
            # Calculate max length threshold from first 3 sub-lines
            sample_lens = [len(s) for s in sub_lines[:3]]
            max_len = max(sample_lens)
            
            # Initial filter by length and nonempty
            filtered_lines = [s.strip() for s in sub_lines if len(s) <= max_len and s.strip()]
            
            # Final filter to drop lines containing forbidden keywords (case-insensitive)
            forbidden = ('test', 'result', 'unit', 'value', 'report')
            filtered_lines = [
                line for line in filtered_lines
                if not any(keyword in line.lower() for keyword in forbidden)
            ]
            
            res = filtered_lines[:-1]
            break
    return res

# ---------------- OCR CONFIDENCE ----------------

def ocr_overall_confidence(image_path: str) -> float:
    img = Image.open(image_path)
    data = pytesseract.image_to_data(img, lang='eng', output_type=Output.DICT)
    confs = []
    for c in data.get('conf', []):
        try:
            v = float(c)
            if v >= 0:
                confs.append(v)
        except ValueError:
            continue
    if not confs:
        return 0.0
    return float(np.mean(confs) / 100.0)
