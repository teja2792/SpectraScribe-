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

One deliberate, documented exception: Modality.UV_VIS_TAUC ("UV-Vis
Tauc") is a SpectraScribe-only enum value with no SpectraVault
equivalent -- see its definition below for why a Tauc plot gets its own
modality tag instead of being folded into plain "UV-Vis".

Scope note (added after this project's first real paper extraction):
SpectraScribe's goal is INTRINSIC MATERIAL PROPERTIES -- composition,
structure, crystal phase, particle/film size and shape, optical bandgap,
and similar -- not reaction or process PERFORMANCE metrics (photo-
catalytic degradation rate, adsorption kinetics, and the like), because
performance numbers are conditional on the exact test setup (light
source, concentration, geometry) in a way a material property mostly
isn't, and mixing the two under one modality/record shape would make
cross-paper comparison misleading. Records that are genuinely reaction-
performance data rather than material characterization (this project's
own Figure 7: photocatalytic dye-degradation kinetics) are kept in the
data -- already-extracted, reviewed work isn't thrown away -- but marked
via the optional out_of_scope/out_of_scope_reason fields below so a
material-properties query or export can exclude them by default.

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
    UV_VIS_TAUC = "UV-Vis Tauc"  # Tauc-transformed UV-Vis data ((alpha*hv)^n vs. hv, for bandgap
                                  # extraction) is a materially different x/y representation of the
                                  # same underlying measurement as plain "UV-Vis" (Absorbance/
                                  # Transmittance vs. Wavelength) -- kept as a distinct modality value,
                                  # not just a different x_axis/y_axis under the same "UV-Vis" tag, per
                                  # explicit user request for specific, searchable modality names
                                  # ("XAS, XANES, UV-Vis Tauc, etc") rather than lumping every UV-Vis-
                                  # derived plot under one generic label. NOT claimed SpectraVault-
                                  # compatible for this one value (see RECORD_SCHEMA's compatibility
                                  # note above) -- same kind of deliberate, documented deviation as
                                  # TABLE_RECORD_SCHEMA's material fields.
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
        # --- SpectraScribe-specific, mandatory (Phase 3 addition) ---
        "material_description", "measurement_purpose",
    ],
    "properties": {
        # --- Core fields, identical meaning to SpectraVault's schema.py ---
        "record_id": {"type": "string"},
        "material_formula": {"type": "string"},
        # material_formula alone (e.g. "TiO2/Cu2O") is not enough to tell
        # two samples in the same paper apart -- this project's own test
        # paper has TiO2/Cu2O coatings made with 10, 30, and 60 minute
        # reaction times, and separate SI figures for each. A curve
        # digitized under just "TiO2/Cu2O" is ambiguous the moment a
        # second paper (or a second figure in the SAME paper) reports a
        # different-process sample with the identical formula.
        # material_description carries whatever process detail the paper
        # itself uses to distinguish samples -- e.g. "TiO2/Cu2O coating,
        # 60 min reaction time, silicon substrate" -- in the paper's own
        # words, not reformatted.
        "material_description": {"type": "string"},
        # What this specific measurement is being COMPARED against, if
        # anything -- e.g. a Raman spectrum of a TiO2/Cu2O sample is
        # meaningless on its own for ruling out substrate artifacts
        # without knowing "compared against a bare-glass-substrate Raman
        # spectrum" is the baseline. Required to be an explicit key (see
        # validate_record()) even when null, so "no clear baseline for
        # this spectrum" is a stated decision, not an omission nobody
        # checked.
        "baseline_material": {"type": ["string", "null"]},
        # Why the paper made this measurement at all -- e.g. "confirm
        # Cu(I) oxidation state via XPS binding energy" or "monitor
        # methyl orange dye concentration during the photocatalysis
        # test." Without this, a digitized curve is numbers with no
        # stated reason to trust what question they were meant to answer.
        "measurement_purpose": {"type": "string"},
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

        # --- Scope flag (optional; absent/false means "in scope") ---
        # See the module docstring's "Scope note" -- this project's goal is
        # intrinsic material properties, not reaction/process performance.
        # Set out_of_scope=true (with a reason) for a record that's genuine,
        # reviewed data but falls outside that goal, so a material-
        # properties query/export can filter it out by default without
        # deleting real work.
        "out_of_scope": {"type": "boolean"},
        "out_of_scope_reason": {"type": ["string", "null"]},
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


TABLE_RECORD_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "title": "SpectraScribe Table Record",
    "type": "object",
    "required": [
        "record_id", "doi", "source_location", "license", "citation",
        "source_type", "extraction_method", "curator", "retrieved_at_utc",
        "table_title", "columns", "rows", "material_description", "measurement_purpose",
    ],
    "properties": {
        "record_id": {"type": "string"},
        "doi": {"type": "string"},
        "source_location": {"type": "string"},  # e.g. "Table S1"
        "license": {"type": "string"},
        "citation": {"type": "string"},
        # material_description/baseline_material/measurement_purpose: same
        # requirement as RECORD_SCHEMA (spectral records) below, extended
        # here to tables for the same reason -- a row of numbers is
        # meaningless without knowing what sample it was measured on, what
        # it's being compared against, and why the measurement was made.
        # First applied to a table for this project's own Figure 7 (kinetics
        # data: 11 curves across 2 panels, each a distinct TiO2/Cu2O sample).
        "material_description": {"type": "string"},
        "baseline_material": {"type": ["string", "null"]},
        "measurement_purpose": {"type": "string"},
        "table_title": {"type": "string"},  # the table's own caption text
        "columns": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["name", "unit"],
                "properties": {"name": {"type": "string"}, "unit": {"type": ["string", "null"]}},
            },
        },
        # Each row is {column_name: value}, where value is either a plain
        # number/string, or {"value": number, "uncertainty": number} when
        # the source table reports a +/- (e.g. "122 +/- 2"), or null when
        # the source table has no entry for that cell (e.g. "-").
        "rows": {"type": "array", "items": {"type": "object"}},
        "source_type": {"enum": [s.value for s in SourceType]},
        "extraction_method": {"enum": [e.value for e in ExtractionMethod]},
        "review_status": {"enum": [r.value for r in ReviewStatus]},
        "curator": {"type": "string"},
        "retrieved_at_utc": {"type": "string", "format": "date-time"},
        "peer_reviewed": {"type": ["boolean", "null"]},
        "open_access": {"type": ["boolean", "null"]},
        "cross_validated": {"type": "boolean"},
        "confidence_score": {"type": ["number", "null"]},
        "confidence_breakdown": {"type": "object"},
        "notes": {"type": "string"},
        # Same scope flag as RECORD_SCHEMA -- see the module docstring's
        # "Scope note". First real use: this project's own Figure 7 table
        # records (photocatalytic degradation kinetics -- reaction
        # performance, not material characterization).
        "out_of_scope": {"type": "boolean"},
        "out_of_scope_reason": {"type": ["string", "null"]},
    },
}
# Deliberately NOT claimed as SpectraVault-compatible, unlike RECORD_SCHEMA
# above -- SpectraVault's own schema has no concept of a generic property
# table (it's built entirely around x_values/y_values spectral curves), so
# there is nothing on the other side of the repo boundary for this shape
# of record to be compatible WITH. Table records are SpectraScribe-only
# for now; if/when Phase 5 (SpectraVault handoff) needs to carry table
# data across, that's a real design question to solve then, not something
# to paper over now by forcing a table into the spectral record shape.
# compute_confidence_score() and its five inputs (source_type,
# extraction_method, peer_reviewed, open_access, cross_validated) are
# reused as-is -- none of those five are curve-shape-specific, so the same
# rubric means the same thing for a table record as for a spectral one.


def validate_table_record(record: dict) -> list:
    """Same pattern as validate_record(), for TABLE_RECORD_SCHEMA. Checks
    structural shape (every row only uses declared column names) in
    addition to the required-field/enum checks, since a transcription
    typo in a row key (e.g. "Layer Thickness" vs "Layer thickness") would
    otherwise silently produce a column no query against this record
    would ever find."""
    problems = []
    record = {k: (v.value if isinstance(v, Enum) else v) for k, v in record.items()}

    for key in TABLE_RECORD_SCHEMA["required"]:
        if key not in record or record[key] in (None, "", []):
            problems.append(f"missing required field: {key}")

    if "baseline_material" not in record:
        problems.append(
            "record has no 'baseline_material' key -- set it to the comparison "
            "sample/measurement if this table is comparative, or explicitly null "
            "if it isn't. Don't omit the key."
        )

    if record.get("source_type") not in [s.value for s in SourceType]:
        problems.append(f"invalid source_type: {record.get('source_type')!r}")
    if record.get("extraction_method") not in [e.value for e in ExtractionMethod]:
        problems.append(f"invalid extraction_method: {record.get('extraction_method')!r}")
    if "review_status" in record and record["review_status"] not in [r.value for r in ReviewStatus]:
        problems.append(f"invalid review_status: {record.get('review_status')!r}")

    column_names = {c["name"] for c in record.get("columns", [])}
    for i, row in enumerate(record.get("rows", [])):
        unknown = set(row.keys()) - column_names
        if unknown:
            problems.append(f"row {i} has keys not declared in columns: {sorted(unknown)}")

    if record.get("doi") and not record.get("source_location"):
        problems.append("record has a doi but no source_location (e.g. 'Table S1') -- not independently checkable")

    if not record.get("open_access"):
        problems.append("record is not marked open_access=True -- SpectraScribe only ingests open-access papers")

    return problems


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

    # baseline_material is intentionally NOT in RECORD_SCHEMA["required"]
    # (its correct value is often legitimately null -- not every spectrum
    # is comparative), but the KEY must always be present. Checked
    # separately from the required-fields loop above, which treats a
    # None value as "missing" -- here None is a valid, meaningful answer
    # ("no baseline for this spectrum"), but an absent key means nobody
    # made that call at all.
    if "baseline_material" not in record:
        problems.append(
            "record has no 'baseline_material' key -- set it to the comparison "
            "sample/spectrum if this measurement is comparative (e.g. a bare-substrate "
            "blank), or explicitly null if it isn't. Don't omit the key."
        )

    x_vals, y_vals = record.get("x_values"), record.get("y_values")
    if isinstance(x_vals, list) and isinstance(y_vals, list) and len(x_vals) != len(y_vals):
        problems.append(f"x_values/y_values length mismatch: {len(x_vals)} vs {len(y_vals)}")

    if not record.get("open_access"):
        # Belt-and-suspenders enforcement of docs/SOURCE_POLICY.md section 1
        # at the schema level, not just as a written policy someone could
        # forget to follow when writing a new ingestion script.
        problems.append("record is not marked open_access=True -- SpectraScribe only ingests open-access papers")

    return problems
