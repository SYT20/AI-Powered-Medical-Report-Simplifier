from pydantic import BaseModel
from typing import List

class RawOutput(BaseModel):
    tests_raw: List[str]
    confidence: float

class ParsedTest(BaseModel):
    name: str
    value: float
    unit: str | None
    status: str

class ParsedOutput(BaseModel):
    tests: List[ParsedTest]
    parse_confidence: float

class SummaryOutput(BaseModel):
    summary: str
    explanations: List[str]

class FinalOutput(BaseModel):
    tests: List[ParsedTest]
    summary: str
    status: str
gemini_client