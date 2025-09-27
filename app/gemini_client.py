import os
import json
import google.generativeai as genai


# Configure API key from environment variable
genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
GEMINI_MODEL = os.getenv('GEMINI_MODEL', 'gemini-2.0-flash')

# Medical test extraction prompt
MEDICAL_EXTRACTION_PROMPT = """---Role---
You are a Medical Data Analysis Specialist who processes raw medical test data and provides structured extraction with patient-friendly summaries.

---Task---
Process the raw OCR medical text in TWO stages:
1. Extract structured medical test data
2. Generate a summary and explanations of the findings

---Stage 1: Medical Test Extraction---
Extract medical tests from the raw text and format as JSON with:
- `name`: Clean test name (e.g., "Blood Urea", "S.Creatinine")
- `value`: Numerical value only
- `unit`: Standardized unit (mg/dL, g/dL, U/L)
- `status`: "normal", "low", or "high" based on reference ranges
- `ref_range`: {{"low": X, "high": Y}} format
- `normalization_confidence`: Computed as (number of tests successfully parsed and normalized ÷ total number of tests in input) × {ocr_confidence}

---Stage 2: Summary Generation---
Create patient-friendly summary with:
- `summary`: Brief, factual statement listing the abnormal findings (focus on abnormal results only)
- `explanations`: Simple, direct explanations of what each abnormal result might indicate (one explanation per abnormal finding)

---Complete Output Format---
Return JSON in this exact structure:
```json
{{
  "extraction": {{
    "tests": [
      {{
        "name": "Test Name",
        "value": numerical_value,
        "unit": "unit_string",
        "status": "normal|low|high",
        "ref_range": {{"low": X, "high": Y}}
      }}
    ],
    "normalization_confidence": 0.XX
  }},
  "summary": {{
    "summary": "Brief factual statement of abnormal findings only",
    "explanations": [
      "Simple explanation of what abnormal finding 1 might indicate",
      "Simple explanation of what abnormal finding 2 might indicate"
    ]
  }}
}}
```

---Examples---

**Example 1:**
Input: "Blood Urea - 28 mg/dl 20-40mg/dl S.Creatinine - 0.8 mg/dl 0.2-1.0mg/dl"
Output:
```json
{{
  "extraction": {{
    "tests": [
      {{
        "name": "Blood Urea",
        "value": 28,
        "unit": "mg/dL",
        "status": "normal",
        "ref_range": {{"low": 20, "high": 40}}
      }},
      {{
        "name": "S.Creatinine",
        "value": 0.8,
        "unit": "mg/dL",
        "status": "normal",
        "ref_range": {{"low": 0.2, "high": 1.0}}
      }}
    ],
    "normalization_confidence": 0.95
  }},
  "summary": {{
    "summary": "All test results are within normal ranges.",
    "explanations": [
      "Normal kidney function tests indicate healthy filtration.",
      "Normal protein levels suggest adequate nutrition and organ function."
    ]
  }}
}}
```

**Example 2:**
Input: "Hemoglobin 10.2 g/dL SGOT - 45U/L 05-35 U/L WBC 11200 /uL"
Output:
```json
{{
  "extraction": {{
    "tests": [
      {{
        "name": "Hemoglobin",
        "value": 10.2,
        "unit": "g/dL",
        "status": "low",
        "ref_range": {{"low": 12.0, "high": 15.0}}
      }},
      {{
        "name": "SGOT",
        "value": 45,
        "unit": "U/L",
        "status": "high",
        "ref_range": {{"low": 5, "high": 35}}
      }},
      {{
        "name": "WBC",
        "value": 11200,
        "unit": "/uL",
        "status": "high",
        "ref_range": {{"low": 4000, "high": 11000}}
      }}
    ],
    "normalization_confidence": 0.88
  }},
  "summary": {{
    "summary": "Low hemoglobin, high SGOT, and high WBC count.",
    "explanations": [
      "Low hemoglobin may relate to anemia or blood loss.",
      "High SGOT can indicate liver cell damage or stress.",
      "High WBC count can occur with infections or inflammation."
    ]
  }}
}}
```

---Safety Rules---
- Never diagnose specific diseases or conditions
- Never suggest treatments or medications  
- Focus on educational explanations only
- Always use patient-friendly language
- Recommend consulting healthcare providers
- Focus ONLY on abnormal findings in summary
- Provide one simple explanation per abnormal result

---Summary Generation Rules---
- **Summary Format**: "Low [test], high [test], and elevated [test]." (list abnormal findings only)
- **Explanation Format**: "[Abnormal finding] may relate to [simple medical explanation]."
- **Focus on Abnormal**: Only mention abnormal results in summary and explanations
- **Simple Language**: Use basic medical terms patients can understand
- **One-to-One**: One explanation per abnormal finding

---Input Text to Process---
```
{input_text}
```

---Output Instructions---
1. Return ONLY valid JSON in the specified format
2. Process both extraction and summary in one response
3. Ensure medical accuracy without diagnosing
4. Use clear, patient-friendly language for explanations
"""

def build_prompt(ocr_text, ocr_confidence):
    """
    Build the medical extraction prompt with input text and OCR confidence
    """
    if isinstance(ocr_text, list):
        input_text = "\n".join(ocr_text)
    else:
        input_text = str(ocr_text)
    return MEDICAL_EXTRACTION_PROMPT.format(input_text=input_text, ocr_confidence=ocr_confidence)

def call_gemini_generate(tests, ocr_confidence):
    prompt = build_prompt(tests, ocr_confidence)
    model = genai.GenerativeModel(GEMINI_MODEL)

    # Generate content using the correct method
    response = model.generate_content(prompt)

    try:
        # Extract JSON from response, removing markdown code blocks if present
        response_text = response.text.strip()
        if response_text.startswith('```json'):
            response_text = response_text[7:]
        if response_text.endswith('```'):
            response_text = response_text[:-3]
        response_text = response_text.strip()

        parsed_response = json.loads(response_text)
        extraction = {
            "tests": parsed_response['extraction']['tests'],
            "normalization_confidence": parsed_response['extraction']['normalization_confidence']
        }
        summary = {
            "summary": parsed_response['summary']['summary'],
            "explanations": parsed_response['summary']['explanations']
        }
        return extraction, summary
    except (json.JSONDecodeError, KeyError) as e:
        print(f"Error parsing Gemini response: {e}")
        print(f"Raw response: {response.text}")
        return {"tests": [], "normalization_confidence": 0.0}, {"summary": "", "explanations": []}
