from dotenv import load_dotenv
load_dotenv() 

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from pathlib import Path

from .ocr import (
    preprocess_medical_report_resize_sharpen,
    extract_medical_test_data,
    ocr_overall_confidence,
    ocr_txt_extraction
)
from .gemini_client import call_gemini_generate

app = FastAPI(title="AI Medical Report OCR")


@app.post("/process-image")
async def process_image(file: UploadFile = File(...)):
    try:
        temp_dir = Path("temp")
        temp_dir.mkdir(exist_ok=True)

        img_path = temp_dir / file.filename
        with open(img_path, "wb") as f:
            f.write(await file.read())

        preprocessed_path = temp_dir / "preprocessed.png"
        preprocess_medical_report_resize_sharpen(str(img_path), str(preprocessed_path))

        ocr_output_txt = ocr_txt_extraction(str(preprocessed_path))
        denoised_ocr_txt = extract_medical_test_data(ocr_output_txt)
        raw_conf = ocr_overall_confidence(str(preprocessed_path))
        raw_out = {"tests_raw": denoised_ocr_txt, "confidence": round(raw_conf, 2)}

        tests,summary = call_gemini_generate(raw_out.get("tests_raw", []), raw_out["confidence"])

        return {
            "step1": denoised_ocr_txt,
            "step2": tests,
            "step3": summary,
            "final": {
                "tests": tests,
                "summary": summary,
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
