"""
schema.py

Canonical record schema for SpectraScribe.

Design goal, stated explicitly: this schema is SpectraVault-compatible,
not just "similar." Every field SpectraVault's own schema.py requires
(record_id, material_formula, modality, x_axis, y_axis, x_values, y_values,
source_type, citation, retrieved_at_utc, source_name, extraction_method,
curator) is required here too, with the same meaning, the same Enum
values, and the same confidence rubric (compute_confidence_score, same
weights). That's deliberate: it's what lets a finished SpectraScribe
record be copied straight into SpectraVault's data/literature_mined/
folder and picked up by SpectraVault's own build_manifest.py /
validate_all.py with zero translation layer -- not a shared code
dependency (SpectraVault is a separate, private repo), just a compatible
file format crossing the repo boundary.

On top of that inherited shape, SpectraScribe adds fields SpectraVault
never needed, because SpectraVault mostly downloads data a database
already agreed to distribute in bulk, while every SpectraScribe record
was extracted from one specific paper's one specific figure or table --
see docs/SOURCE_POLICY.md for why that makes stronger, non-optional
provenance capture the right call here:
  - doi (required): the paper this record came from, full stop. Without
    this, "who wrote this data down first" isn't independently checkable.
  - source_location (required): e.g. "Figure 3b" or "Table 2" -- which
    exact part of the paper. A citation without this still leaves a
    reader hunting through the whole paper to verify one number.

What SpectraScribe records typically look like on the existing rubric:
  source_type is almost always DIGITIZED_FIGURE (0.30 base) -- everything
  here comes from a paper's published figure or table, never a bulk
  database export or an author-supplied raw file. extraction_method is
  usually OCR_EXTRACTED_TABLE (0.08) for numbers read from a reported
  table, or SEMI_AUTOMATED_DIGITIZER (0.10) for a curve traced off a plot
  with human-reviewed confidence -- both already existed in SpectraVault's
  rubric before this project started, because SpectraVault's schema
  anticipated this exact use case (see its ExtractionMethod docstring).
  No new enum values were needed to describe SpectraScribe's own output.
"""

import hashlib
import json
from enum import Enum


class Modality(str, Enum):
    XANES = "XANES"
    XAFS = "XAFS"
    EXAFS = "EXAFS"
    XPS = "XPS"
    UV_VIS = "UV-Vis"
    RAMAN = "Raman"
    FTIR = "FTIR"
    PL = "PL"
    XRD = "XRD"


class SourceType(str, Enum):
    """Kept field-compatible with SpectraVault's SourceType. SpectraScribe
    records will almost always be DIGITIZED_FIGURE -- the other values
    exist only so a record copied between the two repos is never rejected
    for using a value the other schema doesn't recognize."""
    OPEN_DATABASE = "open-database"
    DEPOSITED_RAW = "deposited-raw-file"
    OWN_PUBLISHED_PAPER = "own-published-paper"
    DIGITIZED_FIGURE = "digitized-from-figure"
    HAND_DIGITIZED = "hand-digitized-by-curator"
    COMPUTED_DATABASE = "computed-database"  # kept for schema parity; unused by this pipeline


class ExtractionMethod(str, Enum):
    BULK_DATABASE_DOWNLOAD = "bulk-database-download"
    AUTHOR_SUPPLIED_RAW_FILE = "author-supplied-raw-file"
    SEMI_AUTOMATED_DIGITIZER = "semi-automated-digitizer"  # auto-traced curve, confidence-scored
    HAND_TRACED = "hand-traced"
    OCR_EXTRACTED_TABLE = "ocr-extracted-table"            # numbers read from a reported table
    COMPUTED_FROM_STRUCTURE = "computed-from-structure"    # kept for schema parity; unused here


class ReviewStatus(str, Enum):
    UNREVIEWED = "unreviewed"        # auto-extracted, nobody has confirmed it yet
    SELF_REVIEWED = "self-reviewed"  # a human checked the overlay against the source figure
    CROSS_CHECKED = "cross-checked"  # checked against a second independent source


RECORD_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "title": "SpectraScribe Record",
    "type": "object",
    "required": [
        # --- Inherited from SpectraVault, same meaning ---
        "record_id", "material_formula", "modality", "x_axis", "y_axis",
        "x_values", "y_values", "source_type", "citation", "retrieved_at_utc",
        "source_name", "extraction_method", "curator",
        # --- SpectraScribe-specific, mandatory (see docs/SOURCE_POLICY.md) ---
        "doi", "source_location", "license",
    ],
    "properties": {
        # --- Core fields, identical meaning to SpectraVault's schema.py ---
        "record_id": {"type": "string"},
        "material_formula": {"type": "string"},
        "modality": {"enum": [m.value for m in Modality]},
        "edge": {"type": ["string", "null"]},
        "absorbing_element": {"type": ["string", "null"]},
        "mp_id": {"type": ["string", "null"]},
        "x_axis": {"type": "string"},
        "y_axis": {"type": "string"},
        "x_values": {"type": "array", "items": {"type": "number"}},
        "y_values": {"type": "array", "items": {"type": "number"}},
        "source_type": {"enum": [s.value for s in SourceType]},
        "citation": {"type": "string"},
        "license": {"type": "string"},  # required here, unlike SpectraVault (nullable there)
        "digitization_error_estimate": {"type": ["number", "null"]},
        "linked_properties": {"type": "object"},
        "retrieved_at_utc": {"type": "string", "format": "date-time"},
        "notes": {"type": "string"},

        # --- Provenance ---
        "source_name": {"type": "string"},
        "source_url": {"type": ["string", "null"]},
        "curator": {"type": "string"},
        "content_sha256": {"type": ["string", "null"]},

        # --- SpectraScribe-specific mandatory provenance ---
        "doi": {"type": "string"},
        "source_location": {"type": "string"},  # e.g. "Figure 3b", "Table 2"

        # --- Confidence rubric inputs (identical to SpectraVault) ---
        "peer_reviewed": {"type": ["boolean", "null"]},
        "open_access": {"type": ["boolean", "null"]},
        "cross_validated": {"type": "boolean"},
        "extraction_method": {"enum": [e.value for e in ExtractionMethod]},
        "review_status": {"enum": [r.value for r in ReviewStatus]},

        # --- Confidence rubric output (computed, not hand-entered) ---
        "confidence_score": {"type": ["number", "null"]},
        "confidence_breakdown": {"type": "object"},
    },
}


# Identical weights to SpectraVault's schema.py, on purpose -- a record's
# confidence_score must mean the same thing in both repos, or "filter by
# confidence >= 0.5" stops being a safe operation once records mix.
_SOURCE_TYPE_BASE_SCORE = {
    SourceType.OPEN_DATABASE.value: 0.55,
    SourceType.OWN_PUBLISHED_PAPER.value: 0.55,
    SourceType.DEPOSITED_RAW.value: 0.50,
    SourceType.COMPUTED_DATABASE.value: 0.50,
    SourceType.DIGITIZED_FIGURE.value: 0.30,
    SourceType.HAND_DIGITIZED.value: 0.25,
}
_EXTRACTION_METHOD_SCORE = {
    ExtractionMethod.BULK_DATABASE_DOWNLOAD.value: 0.20,
    ExtractionMethod.AUTHOR_SUPPLIED_RAW_FILE.value: 0.20,
    ExtractionMethod.SEMI_AUTOMATED_DIGITIZER.value: 0.10,
    ExtractionMethod.HAND_TRACED.value: 0.05,
    ExtractionMethod.OCR_EXTRACTED_TABLE.value: 0.08,
    ExtractionMethod.COMPUTED_FROM_STRUCTURE.value: 0.12,
}
_PEER_REVIEWED_BONUS = 0.10
_OPEN_ACCESS_BONUS = 0.05
_CROSS_VALIDATED_BONUS = 0.10


def compute_confidence_score(record: dict) -> tuple:
    """Identical rubric to SpectraVault's compute_confidence_score() --
    same five inputs, same weights, same meaning. Returns (score,
    breakdown); breakdown makes every score auditable rather than a
    black-box number. Deliberately does NOT know whether the spectrum
    itself looks physically reasonable -- that's out of scope for a
    provenance rubric, same as in SpectraVault."""
    breakdown = {}

    source_type = record.get("source_type")
    breakdown["source_type"] = _SOURCE_TYPE_BASE_SCORE.get(source_type, 0.0)

    extraction_method = record.get("extraction_method")
    breakdown["extraction_method"] = _EXTRACTION_METHOD_SCORE.get(extraction_method, 0.0)

    breakdown["peer_reviewed"] = _PEER_REVIEWED_BONUS if record.get("peer_reviewed") else 0.0
    breakdown["open_access"] = _OPEN_ACCESS_BONUS if record.get("open_access") else 0.0
    breakdown["cross_validated"] = _CROSS_VALIDATED_BONUS if record.get("cross_validated") else 0.0

    score = sum(breakdown.values())
    score = max(0.0, min(1.0, score))
    return score, breakdown


def compute_content_hash(x_values: list, y_values: list) -> str:
    """SHA-256 of the (x_values, y_values) pair -- same purpose as
    SpectraVault's version: catching the same curve showing up twice
    (e.g. re-digitized from two different papers that both reproduce the
    same original figure) before it's silently double-counted."""
    payload = json.dumps({"x": x_values, "y": y_values}, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def dump_record(record: dict, path) -> None:
    """Same compact-array writer as SpectraVault's dump_record() -- see
    that project's schema.py docstring for the exact bug this avoids
    (pretty-printing every array element bloats a multi-thousand-point
    digitized curve into an unreadable, slow-to-write file). Kept
    byte-for-byte identical in approach so a record's file size profile
    doesn't change when it crosses from this repo into SpectraVault's."""
    obj = dict(record)
    raw_tokens = {}
    for key in ("x_values", "y_values"):
        if key in obj and isinstance(obj[key], list):
            token = f"__RAW_JSON_{key}__"
            raw_tokens[token] = json.dumps(obj[key], separators=(",", ":"))
            obj[key] = token

    text = json.dumps(obj, indent=2)
    for token, raw in raw_tokens.items():
        text = text.replace(f'"{token}"', raw)

    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def validate_record(record: dict) -> list:
    """Dependency-free validator, same pattern as SpectraVault's. Returns a
    list of problem strings; empty list means the record passes. Enforces
    everything SpectraVault's validator enforces, PLUS the SpectraScribe-
    specific mandatory provenance fields from docs/SOURCE_POLICY.md."""
    problems = []

    record = {k: (v.value if isinstance(v, Enum) else v) for k, v in record.items()}

    for key in RECORD_SCHEMA["required"]:
        if key not in record or record[key] in (None, ""):
            problems.append(f"missing required field: {key}")

    if record.get("modality") not in [m.value for m in Modality]:
        problems.append(f"invalid modality: {record.get('modality')!r}")

    if record.get("source_type") not in [s.value for s in SourceType]:
        problems.append(f"invalid source_type: {record.get('source_type')!r}")

    if record.get("extraction_method") not in [e.value for e in ExtractionMethod]:
        problems.append(f"invalid extraction_method: {record.get('extraction_method')!r}")

    if "review_status" in record and record["review_status"] not in [r.value for r in ReviewStatus]:
        problems.append(f"invalid review_status: {record.get('review_status')!r}")

    if record.get("source_type") in (SourceType.DIGITIZED_FIGURE.value, SourceType.HAND_DIGITIZED.value):
        if record.get("digitization_error_estimate") is None:
            problems.append(
                f"{record.get('source_type')} records must set digitization_error_estimate"
            )

    # SpectraScribe-specific: every record must be traceable to a specific
    # figure/table in a specific paper, not just "cited somewhere."
    if record.get("doi") and not record.get("source_location"):
        problems.append("record has a doi but no source_location (e.g. 'Figure 3b') -- not independently checkable")

    x_vals, y_vals = record.get("x_values"), record.get("y_values")
    if isinstance(x_vals, list) and isinstance(y_vals, list) and len(x_vals) != len(y_vals):
        problems.append(f"x_values/y_values length mismatch: {len(x_vals)} vs {len(y_vals)}")

    if not record.get("open_access"):
        # Belt-and-suspenders enforcement of docs/SOURCE_POLICY.md section 1
        # at the schema level, not just as a written policy someone could
        # forget to follow when writing a new ingestion script.
        problems.append("record is not marked open_access=True -- SpectraScribe only ingests open-access papers")

    return problems
