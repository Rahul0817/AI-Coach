"""Canonical biomarker registry and reference-range interpretation.

Lab reports name the same analyte a dozen ways: "Testosterone, Total",
"TOTAL TESTOSTERONE", "T. Testosterone", "Testo (Total)". Parsing is only
useful if all of those collapse to one canonical key, which is what the alias
table here does.

Reference ranges are stored as **fallbacks only**. When the report prints its
own range — which most do — that one is authoritative, because ranges are
assay- and laboratory-specific. Using a hard-coded range in preference to the
printed one would produce confidently wrong flags, which in a health product is
worse than producing none.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from app.models.enums import BiomarkerFlag


@dataclass(frozen=True, slots=True)
class BiomarkerSpec:
    """Everything known about one lab analyte."""

    key: str
    display_name: str
    aliases: list[str]
    canonical_unit: str
    #: Fallback range, used only when the report does not print its own.
    fallback_low: float | None = None
    fallback_high: float | None = None
    #: Plain-language description of what the marker measures.
    meaning: str = ""
    #: What an above-range value may indicate. Deliberately hedged — this is
    #: education, and only a clinician interprets a result.
    high_note: str = ""
    low_note: str = ""
    #: True for the markers directly relevant to PCOS assessment; these are
    #: surfaced first in the UI.
    pcos_relevant: bool = True
    #: Alternative units and their multiplier to reach ``canonical_unit``.
    unit_conversions: dict[str, float] = field(default_factory=dict)


BIOMARKERS: dict[str, BiomarkerSpec] = {
    "total_testosterone": BiomarkerSpec(
        key="total_testosterone",
        display_name="Total Testosterone",
        aliases=["total testosterone", "testosterone total", "testosterone, total",
                 "testosterone (total)", "t. testosterone", "serum testosterone",
                 "testosterone"],
        canonical_unit="ng/dL",
        fallback_low=15.0, fallback_high=70.0,
        meaning="The total amount of testosterone in your blood, both bound to "
                "proteins and free.",
        high_note="Mildly raised total testosterone is one of the features "
                  "clinicians look at when assessing PCOS. Markedly high values "
                  "prompt investigation of other causes.",
        low_note="Low testosterone is not typical of PCOS and has other causes.",
        unit_conversions={"nmol/l": 28.84, "ng/ml": 100.0},
    ),
    "free_testosterone": BiomarkerSpec(
        key="free_testosterone",
        display_name="Free Testosterone",
        aliases=["free testosterone", "testosterone free", "testosterone, free",
                 "free testo"],
        canonical_unit="pg/mL",
        fallback_low=0.3, fallback_high=3.2,
        meaning="The fraction of testosterone not bound to proteins, and "
                "therefore biologically active.",
        high_note="Free testosterone is often more informative than total in "
                  "PCOS, because high insulin lowers SHBG and raises the free "
                  "fraction even when total testosterone reads normal.",
        low_note="Not typically associated with PCOS.",
    ),
    "shbg": BiomarkerSpec(
        key="shbg",
        display_name="Sex Hormone Binding Globulin (SHBG)",
        aliases=["shbg", "sex hormone binding globulin",
                 "sex hormone-binding globulin"],
        canonical_unit="nmol/L",
        fallback_low=18.0, fallback_high=144.0,
        meaning="The protein that binds testosterone and keeps it inactive.",
        high_note="Higher SHBG means less free androgen circulating.",
        low_note="Low SHBG is itself a recognised marker of insulin resistance, "
                 "and it raises free testosterone even when the total is normal.",
    ),
    "lh": BiomarkerSpec(
        key="lh",
        display_name="Luteinising Hormone (LH)",
        aliases=["lh", "luteinizing hormone", "luteinising hormone",
                 "l.h.", "serum lh"],
        canonical_unit="mIU/mL",
        fallback_low=1.9, fallback_high=12.5,
        meaning="A pituitary hormone whose surge triggers ovulation.",
        high_note="Chronically elevated LH, particularly relative to FSH, is a "
                  "classic PCOS pattern — though it is present in only a "
                  "proportion of cases and is no longer a formal criterion.",
        low_note="Low LH points toward a different mechanism and warrants "
                 "clinical assessment.",
    ),
    "fsh": BiomarkerSpec(
        key="fsh",
        display_name="Follicle-Stimulating Hormone (FSH)",
        aliases=["fsh", "follicle stimulating hormone",
                 "follicle-stimulating hormone", "f.s.h.", "serum fsh"],
        canonical_unit="mIU/mL",
        fallback_low=2.5, fallback_high=10.2,
        meaning="A pituitary hormone that drives follicle development.",
        high_note="Raised FSH is assessed in the context of age and cycle day.",
        low_note="Interpreted alongside LH; the ratio matters more than either alone.",
    ),
    "amh": BiomarkerSpec(
        key="amh",
        display_name="Anti-Müllerian Hormone (AMH)",
        aliases=["amh", "anti mullerian hormone", "anti-müllerian hormone",
                 "anti-mullerian hormone"],
        canonical_unit="ng/mL",
        fallback_low=1.0, fallback_high=4.0,
        meaning="Reflects the number of small follicles in the ovaries.",
        high_note="AMH is typically elevated in PCOS because there are more "
                  "small follicles. It indicates follicle count, not fertility.",
        low_note="Lower AMH reflects a smaller follicle pool.",
    ),
    "prolactin": BiomarkerSpec(
        key="prolactin",
        display_name="Prolactin",
        aliases=["prolactin", "prl", "serum prolactin"],
        canonical_unit="ng/mL",
        fallback_low=4.8, fallback_high=23.3,
        meaning="A pituitary hormone; measured mainly to rule out other causes "
                "of irregular cycles.",
        high_note="Raised prolactin causes irregular periods through a different "
                  "mechanism than PCOS and is treated differently, so it is "
                  "checked as part of excluding other explanations.",
        low_note="Low prolactin is rarely clinically significant on its own.",
    ),
    "tsh": BiomarkerSpec(
        key="tsh",
        display_name="Thyroid-Stimulating Hormone (TSH)",
        aliases=["tsh", "thyroid stimulating hormone", "thyroid-stimulating hormone"],
        canonical_unit="µIU/mL",
        fallback_low=0.4, fallback_high=4.0,
        meaning="The primary screening test for thyroid function.",
        high_note="A raised TSH suggests an underactive thyroid, which produces "
                  "symptoms that overlap heavily with PCOS and is treatable.",
        low_note="A low TSH suggests an overactive thyroid.",
        unit_conversions={"miu/l": 1.0, "µiu/ml": 1.0, "uiu/ml": 1.0},
    ),
    "dhea_s": BiomarkerSpec(
        key="dhea_s",
        display_name="DHEA-Sulphate (DHEA-S)",
        aliases=["dhea-s", "dhea s", "dheas", "dhea sulfate", "dhea sulphate",
                 "dehydroepiandrosterone sulfate"],
        canonical_unit="µg/dL",
        fallback_low=35.0, fallback_high=430.0,
        meaning="An androgen produced by the adrenal glands.",
        high_note="Moderately raised DHEA-S occurs in a subset of PCOS. Markedly "
                  "high values point toward an adrenal source and need specialist "
                  "assessment.",
        low_note="Low DHEA-S is not associated with PCOS.",
    ),
    "fasting_glucose": BiomarkerSpec(
        key="fasting_glucose",
        display_name="Fasting Glucose",
        aliases=["fasting glucose", "glucose fasting", "fasting blood glucose",
                 "fbs", "fasting blood sugar", "glucose, fasting"],
        canonical_unit="mg/dL",
        fallback_low=70.0, fallback_high=99.0,
        meaning="Blood sugar after an overnight fast.",
        high_note="Values above the reference range are how prediabetes and "
                  "diabetes are screened for. Note that fasting glucose is often "
                  "normal even when insulin resistance is significant.",
        low_note="Low fasting glucose has several causes and is assessed clinically.",
        unit_conversions={"mmol/l": 18.0182},
    ),
    "fasting_insulin": BiomarkerSpec(
        key="fasting_insulin",
        display_name="Fasting Insulin",
        aliases=["fasting insulin", "insulin fasting", "insulin, fasting",
                 "serum insulin"],
        canonical_unit="µIU/mL",
        fallback_low=2.6, fallback_high=24.9,
        meaning="How much insulin the pancreas is producing at rest.",
        high_note="Raised fasting insulin is a more sensitive early marker of "
                  "insulin resistance than glucose, because the pancreas "
                  "compensates by producing more before glucose ever rises.",
        low_note="Interpreted alongside glucose.",
    ),
    "hba1c": BiomarkerSpec(
        key="hba1c",
        display_name="HbA1c",
        aliases=["hba1c", "hb a1c", "glycated haemoglobin", "glycated hemoglobin",
                 "glycosylated hemoglobin", "a1c"],
        canonical_unit="%",
        fallback_low=4.0, fallback_high=5.6,
        meaning="Average blood sugar over roughly the past three months.",
        high_note="5.7–6.4% is the range generally described as prediabetes and "
                  "6.5% or above as diabetes. Your clinician interprets this "
                  "alongside your other results.",
        low_note="Low HbA1c is uncommon and assessed clinically.",
    ),
    "homa_ir": BiomarkerSpec(
        key="homa_ir",
        display_name="HOMA-IR",
        aliases=["homa-ir", "homa ir", "homair", "homa index",
                 "insulin resistance index"],
        canonical_unit="index",
        fallback_low=0.0, fallback_high=2.0,
        meaning="An index of insulin resistance calculated from fasting glucose "
                "and fasting insulin.",
        high_note="Higher values indicate greater insulin resistance. Thresholds "
                  "vary between populations and laboratories.",
        low_note="Lower values indicate better insulin sensitivity.",
    ),
    "total_cholesterol": BiomarkerSpec(
        key="total_cholesterol",
        display_name="Total Cholesterol",
        aliases=["total cholesterol", "cholesterol total", "cholesterol, total",
                 "s. cholesterol"],
        canonical_unit="mg/dL",
        fallback_low=0.0, fallback_high=200.0,
        meaning="All cholesterol carried in your blood.",
        high_note="Assessed as part of overall cardiovascular risk, which is "
                  "elevated in PCOS.",
        low_note="Very low total cholesterol is uncommon.",
        pcos_relevant=False,
        unit_conversions={"mmol/l": 38.67},
    ),
    "hdl": BiomarkerSpec(
        key="hdl",
        display_name="HDL Cholesterol",
        aliases=["hdl", "hdl cholesterol", "hdl-c", "cholesterol hdl"],
        canonical_unit="mg/dL",
        fallback_low=50.0, fallback_high=100.0,
        meaning="The cholesterol fraction generally regarded as protective.",
        high_note="Higher HDL is generally favourable.",
        low_note="Low HDL together with high triglycerides is the pattern "
                 "characteristically seen with insulin resistance.",
        unit_conversions={"mmol/l": 38.67},
    ),
    "ldl": BiomarkerSpec(
        key="ldl",
        display_name="LDL Cholesterol",
        aliases=["ldl", "ldl cholesterol", "ldl-c", "cholesterol ldl"],
        canonical_unit="mg/dL",
        fallback_low=0.0, fallback_high=100.0,
        meaning="The cholesterol fraction associated with arterial plaque.",
        high_note="Target levels depend on your overall cardiovascular risk.",
        low_note="Low LDL is generally favourable.",
        pcos_relevant=False,
        unit_conversions={"mmol/l": 38.67},
    ),
    "triglycerides": BiomarkerSpec(
        key="triglycerides",
        display_name="Triglycerides",
        aliases=["triglycerides", "triglyceride", "tg", "s. triglycerides"],
        canonical_unit="mg/dL",
        fallback_low=0.0, fallback_high=150.0,
        meaning="A type of fat carried in the blood.",
        high_note="Raised triglycerides with low HDL is a characteristic pattern "
                  "in insulin resistance.",
        low_note="Low triglycerides are generally not a concern.",
        unit_conversions={"mmol/l": 88.57},
    ),
    "vitamin_d": BiomarkerSpec(
        key="vitamin_d",
        display_name="Vitamin D (25-OH)",
        aliases=["vitamin d", "25-oh vitamin d", "25 oh vitamin d",
                 "25-hydroxyvitamin d", "vit d", "vitamin d3"],
        canonical_unit="ng/mL",
        fallback_low=30.0, fallback_high=100.0,
        meaning="Your vitamin D status.",
        high_note="Very high levels usually reflect over-supplementation.",
        low_note="Deficiency is common in PCOS and is worth correcting — "
                 "discuss dosing with your doctor rather than self-prescribing.",
        unit_conversions={"nmol/l": 0.4},
    ),
}

#: Reverse index from every alias to its canonical key, longest alias first so
#: "total testosterone" wins over the bare "testosterone" substring.
ALIAS_INDEX: list[tuple[str, str]] = sorted(
    ((alias.lower(), spec.key) for spec in BIOMARKERS.values() for alias in spec.aliases),
    key=lambda pair: len(pair[0]),
    reverse=True,
)


def resolve_alias(label: str) -> str | None:
    """Map a printed label onto a canonical biomarker key."""
    normalised = " ".join(label.lower().replace(":", " ").split())
    for alias, key in ALIAS_INDEX:
        if alias in normalised:
            return key
    return None


#: ASCII spellings labs use for the micro prefix. Normalising these first means
#: "uIU/mL", "mcg/dL" and "µIU/mL" are recognised as the same unit everywhere,
#: rather than each biomarker needing its own alias for the variants.
_MICRO_VARIANTS = (("μ", "µ"), ("mc", "µ"), ("u", "µ"))


def _normalise_unit(unit: str) -> str:
    """Lowercase a unit and fold the micro-prefix spellings together."""
    text = unit.lower().strip().replace("μ", "µ")
    for prefix, canonical in _MICRO_VARIANTS:
        if text.startswith(prefix) and not text.startswith(canonical):
            candidate = canonical + text[len(prefix) :]
            # Only rewrite when it produces a real unit, so "ug" becomes "µg"
            # but "unit" and "mmol/l" are left alone.
            if candidate.startswith(("µg", "µiu", "µmol", "µl")):
                return candidate
    return text


def convert_unit(spec: BiomarkerSpec, value: float, unit: str) -> tuple[float, str]:
    """Convert a value into the biomarker's canonical unit where possible."""
    normalised = _normalise_unit(unit)
    if not normalised or normalised == spec.canonical_unit.lower():
        return value, spec.canonical_unit
    multiplier = spec.unit_conversions.get(normalised)
    if multiplier is None:
        # Unknown unit — keep the value and the unit as printed rather than
        # silently mis-scaling it.
        return value, unit
    return round(value * multiplier, 3), spec.canonical_unit


def classify(
    value: float, low: float | None, high: float | None
) -> BiomarkerFlag:
    """Flag a value against a reference range."""
    if low is None and high is None:
        return BiomarkerFlag.UNKNOWN
    if low is not None and value < low:
        return BiomarkerFlag.LOW
    if high is not None and value > high:
        return BiomarkerFlag.HIGH
    return BiomarkerFlag.NORMAL


def interpret(spec: BiomarkerSpec, flag: BiomarkerFlag, value: float, unit: str) -> str:
    """Compose the plain-language explanation shown next to a result."""
    parts = [spec.meaning]

    if flag == BiomarkerFlag.HIGH:
        parts.append(f"Your result of {value} {unit} is above the reference range "
                     f"printed on this report. {spec.high_note}")
    elif flag == BiomarkerFlag.LOW:
        parts.append(f"Your result of {value} {unit} is below the reference range "
                     f"printed on this report. {spec.low_note}")
    elif flag == BiomarkerFlag.NORMAL:
        parts.append(f"Your result of {value} {unit} falls within the reference "
                     f"range printed on this report.")
    else:
        parts.append(f"Your result is {value} {unit}. No reference range was "
                     f"found on the report, so this value cannot be placed in "
                     f"context here.")

    parts.append(
        "Reference ranges differ between laboratories and assays, and a single "
        "value is never interpreted in isolation. Your doctor reads this "
        "alongside your symptoms and your other results."
    )
    return " ".join(p for p in parts if p)
