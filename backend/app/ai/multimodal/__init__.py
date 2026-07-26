"""Multimodal input: OCR for lab reports, vision for meals, and voice."""

from app.ai.multimodal.biomarkers import BIOMARKERS, BiomarkerSpec, resolve_alias
from app.ai.multimodal.nutrition_db import FOODS, FoodItem, find_all_foods, score_meal
from app.ai.multimodal.ocr import OCRResult, parse_biomarkers, process_report
from app.ai.multimodal.vision import build_analysis_payload, get_vision_provider
from app.ai.multimodal.voice import capabilities as voice_capabilities

__all__ = [
    "BIOMARKERS",
    "FOODS",
    "BiomarkerSpec",
    "FoodItem",
    "OCRResult",
    "build_analysis_payload",
    "find_all_foods",
    "get_vision_provider",
    "parse_biomarkers",
    "process_report",
    "resolve_alias",
    "score_meal",
    "voice_capabilities",
]
