"""Blood-report text extraction and biomarker parsing.

Extraction strategy, cheapest first
-----------------------------------
1. **Digital PDF → embedded text layer** (``pypdf``). Most lab reports emailed
   to patients are digitally generated and already contain perfect text. Running
   OCR on them would be slower *and* less accurate — OCR introduces errors that
   are simply not present in the text layer.
2. **Scanned PDF → rasterise, then OCR.** Only when step 1 yields too little
   text to be a real report.
3. **Image → OCR** via Tesseract.

Reporting confidence honestly matters here. Tesseract exposes per-word
confidence, and it is surfaced rather than hidden, because a user whose report
was parsed at 40% confidence needs to know the numbers may be wrong.

Parsing
-------
The parser handles the line layouts real reports use — value and range on the
same line, range on the following line, ranges written as ``12 - 45``,
``< 150``, or ``Ref: 0.4-4.0``. It is regex-based rather than model-based
because the output feeds a health explanation, and a deterministic parser that
misses a line is far safer than a model that invents one.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from dataclasses import dataclass
from io import BytesIO

from app.ai.multimodal.biomarkers import (
    BIOMARKERS,
    classify,
    convert_unit,
    interpret,
    resolve_alias,
)
from app.core.config import settings
from app.core.exceptions import UnsupportedMediaError
from app.core.logging import get_logger
from app.models.enums import BiomarkerFlag

logger = get_logger(__name__)

SUPPORTED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp", "image/tiff"}
SUPPORTED_TYPES = SUPPORTED_IMAGE_TYPES | {"application/pdf"}

#: Below this many characters, a PDF's text layer is assumed to be absent or
#: decorative and the scanned-document path is taken instead.
MIN_TEXT_LAYER_CHARS = 180


@dataclass(slots=True)
class ParsedBiomarker:
    key: str
    display_name: str
    value: float
    unit: str
    reference_low: float | None
    reference_high: float | None
    flag: BiomarkerFlag
    interpretation: str
    raw_line: str


@dataclass(slots=True)
class OCRResult:
    text: str
    confidence: float | None
    method: str
    biomarkers: list[ParsedBiomarker]
    warnings: list[str]


# A numeric value, optionally signed/decimal.
_NUMBER = r"[-+]?\d+(?:[.,]\d+)?"
#: Units seen on real reports. Ordered so multi-character units match first.
_UNIT = (
    r"(?:ng/dL|ng/mL|pg/mL|µg/dL|ug/dL|mcg/dL|mIU/mL|µIU/mL|uIU/mL|mIU/L|"
    r"IU/L|nmol/L|µmol/L|umol/L|mmol/L|mg/dL|mg/L|g/dL|%|ratio|index)"
)

#: value + unit + an inline reference range on the same line
_RANGE_PATTERNS = [
    # "12.5 - 45.0" or "12.5 to 45.0"
    re.compile(rf"({_NUMBER})\s*(?:-|–|—|to)\s*({_NUMBER})"),
    # "Ref: 0.4-4.0" / "Reference Range 0.4 - 4.0"
    re.compile(
        rf"(?:ref(?:erence)?(?:\s*range)?|normal|biological\s*ref)\s*[:.]?\s*"
        rf"({_NUMBER})\s*(?:-|–|—|to)\s*({_NUMBER})",
        re.I,
    ),
]
_UPPER_ONLY = re.compile(rf"(?:<|less than|upto|up to)\s*({_NUMBER})", re.I)
_LOWER_ONLY = re.compile(rf"(?:>|greater than|above)\s*({_NUMBER})", re.I)

_VALUE_UNIT = re.compile(rf"({_NUMBER})\s*({_UNIT})", re.I)


def _to_float(raw: str) -> float | None:
    try:
        return float(raw.replace(",", "."))
    except ValueError:
        return None


def extract_text_from_pdf(data: bytes) -> tuple[str, str, list[str]]:
    """Read a PDF's embedded text layer. Returns ``(text, method, warnings)``."""
    warnings: list[str] = []
    try:
        from pypdf import PdfReader

        reader = PdfReader(BytesIO(data))
        pages = [page.extract_text() or "" for page in reader.pages[:20]]
        text = "\n".join(pages).strip()
        if len(text) >= MIN_TEXT_LAYER_CHARS:
            return text, "pdf_text_layer", warnings
        warnings.append(
            "This PDF has little or no embedded text, so it was treated as a scan."
        )
    except Exception as exc:
        warnings.append(f"Could not read the PDF text layer: {exc}")

    # Scanned PDF: rasterise then OCR.
    try:
        from pdf2image import convert_from_bytes

        images = convert_from_bytes(data, dpi=250, first_page=1, last_page=5)
        chunks: list[str] = []
        for image in images:
            page_text, _ = _tesseract_image(image)
            chunks.append(page_text)
        return "\n".join(chunks).strip(), "pdf_ocr", warnings
    except Exception as exc:
        warnings.append(
            "This appears to be a scanned PDF, and the tools needed to read it "
            f"(poppler + Tesseract) are not available: {exc}"
        )
        return "", "pdf_failed", warnings


def _tesseract_image(image) -> tuple[str, float | None]:  # type: ignore[no-untyped-def]
    """Run Tesseract over a PIL image, returning text and mean confidence."""
    import pytesseract

    if settings.tesseract_cmd and settings.tesseract_cmd != "tesseract":
        pytesseract.pytesseract.tesseract_cmd = settings.tesseract_cmd

    text = pytesseract.image_to_string(image)
    confidence: float | None = None
    try:
        data = pytesseract.image_to_data(
            image, output_type=pytesseract.Output.DICT
        )
        scores = [int(c) for c in data.get("conf", []) if str(c).lstrip("-").isdigit()]
        # Tesseract emits -1 for non-text regions; including them would drag the
        # reported confidence down for no reason.
        scores = [s for s in scores if s >= 0]
        if scores:
            confidence = round(sum(scores) / len(scores) / 100, 3)
    except Exception:
        pass
    return text, confidence


def extract_text_from_image(data: bytes) -> tuple[str, float | None, list[str]]:
    """OCR an image file."""
    warnings: list[str] = []
    if shutil.which(settings.tesseract_cmd) is None:
        warnings.append(
            "Tesseract OCR is not installed on this server, so text could not "
            "be read from the image. Upload a digital PDF instead, or enter the "
            "values manually."
        )
        return "", None, warnings

    try:
        from PIL import Image

        image = Image.open(BytesIO(data))
        # Greyscale improves OCR accuracy on printed lab reports noticeably.
        if image.mode not in {"L", "RGB"}:
            image = image.convert("RGB")
        text, confidence = _tesseract_image(image.convert("L"))
        return text.strip(), confidence, warnings
    except Exception as exc:
        warnings.append(f"Could not read this image: {exc}")
        return "", None, warnings


def parse_biomarkers(text: str) -> list[ParsedBiomarker]:
    """Extract structured biomarker values from raw report text."""
    results: dict[str, ParsedBiomarker] = {}
    lines = [line.strip() for line in text.splitlines() if line.strip()]

    for index, line in enumerate(lines):
        key = resolve_alias(line)
        if key is None:
            continue
        spec = BIOMARKERS[key]

        # Real reports sometimes wrap a result onto the following line, so the
        # next line is considered part of this record.
        window = line
        if index + 1 < len(lines) and not resolve_alias(lines[index + 1]):
            window = f"{line} {lines[index + 1]}"

        value_match = _VALUE_UNIT.search(window)
        if value_match:
            raw_value = _to_float(value_match.group(1))
            unit = value_match.group(2)
        else:
            # No unit printed. Take the first standalone number after the label.
            after_label = window[value_match.end():] if value_match else window
            numbers = re.findall(_NUMBER, after_label or window)
            raw_value = _to_float(numbers[0]) if numbers else None
            unit = spec.canonical_unit

        if raw_value is None:
            continue

        original_unit = unit
        value, unit = convert_unit(spec, raw_value, unit)

        # --- reference range, preferring the one printed on the report ---
        low = high = None
        # Search after the measured value so the value itself is not mistaken
        # for the lower bound of its own range.
        tail = window[value_match.end():] if value_match else window
        for pattern in _RANGE_PATTERNS:
            range_match = pattern.search(tail)
            if range_match:
                low = _to_float(range_match.group(1))
                high = _to_float(range_match.group(2))
                break
        if low is None and high is None:
            if upper := _UPPER_ONLY.search(tail):
                high = _to_float(upper.group(1))
            elif lower := _LOWER_ONLY.search(tail):
                low = _to_float(lower.group(1))

        # The printed range is expressed in the unit printed on the report, so
        # it must ride through the same conversion as the value. Skipping this
        # compares a converted result against an unconverted range, which
        # silently produces wrong HIGH/LOW flags — for example a testosterone
        # of 2.9 nmol/L (range 0.5–2.4) becoming 83.6 ng/dL judged against 0.5–2.4.
        if low is not None or high is not None:
            if low is not None:
                low, _ = convert_unit(spec, low, original_unit)
            if high is not None:
                high, _ = convert_unit(spec, high, original_unit)

        source_note = ""
        if low is None and high is None:
            low, high = spec.fallback_low, spec.fallback_high
            source_note = (
                " No reference range was printed on the report, so a typical "
                "range is shown for context only — check it against your own "
                "report."
            )

        flag = classify(value, low, high)
        results[key] = ParsedBiomarker(
            key=key,
            display_name=spec.display_name,
            value=value,
            unit=unit,
            reference_low=low,
            reference_high=high,
            flag=flag,
            interpretation=interpret(spec, flag, value, unit) + source_note,
            raw_line=line[:200],
        )

    # PCOS-relevant markers first, then alphabetically for stable ordering.
    return sorted(
        results.values(),
        key=lambda b: (not BIOMARKERS[b.key].pcos_relevant, b.display_name),
    )


def process_report(data: bytes, content_type: str) -> OCRResult:
    """Full pipeline: extract text, then parse biomarkers from it."""
    if content_type not in SUPPORTED_TYPES:
        raise UnsupportedMediaError(
            f"'{content_type}' is not supported. Upload a PDF, JPEG, PNG, "
            f"WebP or TIFF file."
        )

    if content_type == "application/pdf":
        text, method, warnings = extract_text_from_pdf(data)
        confidence = 1.0 if method == "pdf_text_layer" else None
    else:
        text, confidence, warnings = extract_text_from_image(data)
        method = "image_ocr"

    biomarkers = parse_biomarkers(text) if text else []

    if text and not biomarkers:
        warnings.append(
            "Text was extracted, but no recognised biomarkers were found. The "
            "report may use an unusual layout — you can enter values manually."
        )
    if confidence is not None and confidence < 0.6:
        warnings.append(
            f"Text recognition confidence was low ({confidence:.0%}). Please "
            f"check every value against your original report."
        )

    logger.info(
        "report processed",
        extra={
            "method": method,
            "text_chars": len(text),
            "biomarkers": len(biomarkers),
            "confidence": confidence,
        },
    )
    return OCRResult(
        text=text,
        confidence=confidence,
        method=method,
        biomarkers=biomarkers,
        warnings=warnings,
    )


def tesseract_available() -> bool:
    """Whether the Tesseract binary is on PATH — reported by the health check."""
    return shutil.which(settings.tesseract_cmd) is not None


def tesseract_version() -> str | None:  # pragma: no cover - environment dependent
    if not tesseract_available():
        return None
    try:
        output = subprocess.run(
            [settings.tesseract_cmd, "--version"],
            capture_output=True, text=True, timeout=5, check=False,
        )
        return output.stdout.splitlines()[0] if output.stdout else None
    except Exception:
        return None
