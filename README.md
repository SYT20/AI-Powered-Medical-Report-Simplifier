# Plum2: AI-Powered Medical Report OCR

Plum2 is a FastAPI-based web service that uses optical character recognition (OCR) and Google's Gemini AI to automatically extract and interpret medical test results from scanned images of medical reports. It simplifies complex medical data into clear, structured information and patient-friendly summaries.

## Features

- **Image Upload**: Accepts medical report images (JPEG, PNG, etc.)
- **OCR Processing**: Extracts text from images using Tesseract OCR with advanced image preprocessing
- **Data Extraction**: Parses raw OCR text to identify medical test results
- **AI Analysis**: Uses Gemini AI to structure test data and generate summaries
- **Confidence Scoring**: Provides confidence levels for OCR accuracy
- **Patient-Friendly Output**: Creates easy-to-understand explanations of results

## Architecture

The application follows a modular architecture with three main processing stages:

### Core Components

1. **FastAPI Server** (`main.py`)
   - Handles HTTP requests and file uploads
   - Coordinates the entire processing pipeline
   - Returns structured JSON responses

2. **OCR Module** (`ocr.py`)
   - **Preprocessing**: Enhances images for better text recognition (resizing, sharpening, thresholding)
   - **Text Extraction**: Converts images to text using Tesseract OCR
   - **Data Parsing**: Filters and extracts medical test information from raw text
   - **Confidence Calculation**: Measures OCR accuracy

3. **Gemini AI Client** (`gemini_client.py`)
   - **Structured Extraction**: Converts parsed text into standardized medical test format
   - **Summary Generation**: Creates patient-friendly explanations
   - **Medical Analysis**: Determines if results are normal, low, or high

4. **Data Models** (`models.py`)
   - Defines structured data formats using Pydantic
   - Ensures type safety and validation

### Processing Pipeline

```
Image Upload → Preprocessing → OCR Text Extraction → Data Filtering → AI Analysis → Structured Output
```

## Prerequisites

Before setting up the project, ensure you have:

- **Python 3.8+** installed
- **Tesseract OCR** installed on your system
- **Google Gemini API key** (for AI analysis)
- Basic knowledge of running web applications

### Installing Tesseract OCR

**Windows:**
- Download from [GitHub releases](https://github.com/UB-Mannheim/tesseract/wiki)
- Add to system PATH or set `TESSERACT_CMD` environment variable

**macOS:**
```bash
brew install tesseract
```

**Linux (Ubuntu/Debian):**
```bash
sudo apt-get install tesseract-ocr
```

## Setup Instructions

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/plum2.git
   cd plum2
   ```

2. **Create virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables:**

   Create a `.env` file in the project root:
   ```
   GEMINI_API_KEY=your_gemini_api_key_here
   GEMINI_MODEL=gemini-2.0-flash
   TESSERACT_CMD=/path/to/tesseract  # Optional, if not in PATH
   ```

   Get your Gemini API key from [Google AI Studio](https://aistudio.google.com/).

5. **Run the application:**
   ```bash
   uvicorn app.main:app --reload
   ```

   The API will be available at `http://localhost:8000`

6. **Verify installation:**

   Visit `http://localhost:8000/docs` to see the interactive API documentation.

## API Usage

### Endpoint: `/process-image`

Processes a medical report image and returns structured test results.

**Method:** `POST`  
**Content-Type:** `multipart/form-data`

**Parameters:**
- `file` (required): Image file (JPEG, PNG, etc.)

**Response Format:**
```json
{
  "step1": ["Raw extracted test lines"],
  "step2": {
    "tests": [
      {
        "name": "Test Name",
        "value": 28.0,
        "unit": "mg/dL",
        "status": "normal",
        "ref_range": {"low": 20, "high": 40}
      }
    ],
    "normalization_confidence": 0.95
  },
  "step3": {
    "summary": "Brief summary of findings",
    "explanations": ["Patient-friendly explanations"]
  },
  "final": {
    "tests": [...],
    "summary": "..."
  }
}
```

## End-to-End Example

Let's walk through processing a sample medical report image.

### 1. Prepare Sample Data

The project includes sample data in the `Data/` folder. For this example, we'll use `img2.png`.

### 2. Start the Server

Make sure the server is running:
```bash
uvicorn app.main:app --reload
```

### 3. Send Image via API

Using curl:
```bash
curl -X POST "http://localhost:8000/process-image" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@Data/img2.png"
```

Using Python requests:
```python
import requests

url = "http://localhost:8000/process-image"
files = {"file": open("Data/img2.png", "rb")}
response = requests.post(url, files=files)
result = response.json()

print("Raw extracted data:", result["step1"])
print("Structured tests:", result["step2"]["tests"])
print("Summary:", result["step3"]["summary"])
```

### 4. Understanding the Output

The API returns results in stages:

- **step1**: Raw lines extracted from the image after OCR processing
- **step2**: Structured test data with normalization confidence
- **step3**: AI-generated summary and explanations
- **final**: Combined final output

### 5. Sample Output

```json
{
  "step1": [
    "Blood Urea - 28 mg/dl 20-40mg/dl",
    "S.Creatinine - 0.8 mg/dl 0.2-1.0mg/dl"
  ],
  "step2": {
    "tests": [
      {
        "name": "Blood Urea",
        "value": 28.0,
        "unit": "mg/dL",
        "status": "normal",
        "ref_range": {"low": 20, "high": 40}
      },
      {
        "name": "S.Creatinine",
        "value": 0.8,
        "unit": "mg/dL",
        "status": "normal",
        "ref_range": {"low": 0.2, "high": 1.0}
      }
    ],
    "normalization_confidence": 0.95
  },
  "step3": {
    "summary": "All test results are within normal ranges.",
    "explanations": [
      "Normal kidney function tests indicate healthy filtration."
    ]
  },
  "final": {
    "tests": [...],
    "summary": "..."
  }
}
```

### 6. Error Handling

If processing fails, the API returns HTTP 500 with error details:
```json
{
  "detail": "Error description"
}
```

Common issues:
- Invalid image format
- OCR failure (poor image quality)
- Gemini API errors (check API key and quota)

## Troubleshooting

### OCR Issues
- Ensure Tesseract is properly installed and in PATH
- Check image quality (resolution, contrast)
- The preprocessing step handles most common issues automatically

### API Key Issues
- Verify your Gemini API key is valid and has sufficient quota
- Check `.env` file is in the project root
- Ensure the key has access to the specified model

### Performance
- Processing time depends on image size and complexity
- Typical processing: 5-15 seconds per image
- Temporary files are stored in `temp/` folder

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes and test thoroughly
4. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Disclaimer

This tool is for educational and informational purposes only. It does not provide medical advice, diagnosis, or treatment. Always consult qualified healthcare professionals for medical interpretation and decisions.