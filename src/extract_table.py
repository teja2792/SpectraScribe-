"""
extract_table.py

Phase 2, part 2: turn a detected table region (currently just a cropped
PNG -- see ingest_paper.py) into an actual structured record: columns,
rows, values a person or script can query, not pixels someone has to
read by eye every time.

This script deliberately does NOT do its own OCR. Two reasons, not one:

  1. Table transcription is exactly the kind of "looks right but is
     silently wrong" failure mode this project has repeatedly designed
     around (see README: "not blind automation"). Merged header cells,
     +/- uncertainty notation, units-in-header-vs-units-in-cell, and
     em-dash-for-"not measured" are all things a generic OCR pass gets
     wrong in ways that produce a plausible-looking WRONG number, not an
     obvious error. A wrong number that looks right is worse than no
     number.
  2. The actual transcription is well within reach of a careful
     read-the-image pass (by a human, or an LLM that can see the crop) --
     this project already has that available. What's missing isn't
     transcription capability, it's a structured place to PUT the
     transcription with its provenance attached and validated before
     it's trusted.

So the division of labor here is: something else (a human, or an
LLM reading the crop) produces a small JSON file -- just {"columns": [...],
"rows": [...]}, the content of the table and nothing else -- and this
script's job is to attach everything a raw transcription doesn't have on
its own: which paper, which exact table, what the citation is, what
license applies, a computed confidence score, and a review_status that
starts at "unreviewed" and has to be explicitly promoted, the same
checkpoint pattern used for citation metadata in fetch_metadata.py.

Transcription file format (see docs/DATA_LAYOUT.md for a worked example):
    {
      "columns": [
        {"name": "Sample", "unit": null},
        {"name": "Layer thickness", "unit": "nm"}
      ],
      "rows": [
        {"Sample": "TiO2", "Layer thickness": {"value": 122, "uncertainty": 2}},
        {"Sample": "compact silica", "Layer thickness": {"value": 208, "uncertainty": 3}}
      ]
    }
A cell value is a plain number/string, a {"value", "uncertainty"} object
for a reported +/-, or null for a source table's own "-" (not measured).

Usage:
    python extract_table.py --doi 10.xxxx/yyyy --label "Table S1" \
        --transcription table_s1_transcription.json --data-root ../data
    python extract_table.py --doi 10.xxxx/yyyy --label "Table S1" --mark-reviewed
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from ingest_paper import paper_id_from_doi
from schema import (
    ExtractionMethod,
    ReviewStatus,
    SourceType,
    compute_confidence_score,
    validate_table_record,
)


def _find_region(manifest: dict, label: str) -> dict:
    normalized = label.lower().replace(" ", "")
    for region in manifest["regions"]:
        if region["label"].lower().replace(" ", "") == normalized:
            return region
    available = [r["label"] for r in manifest["regions"] if r["kind"] == "table"]
    raise ValueError(f"No region labeled {label!r} in this paper's manifest. Tables detected: {available}")


def extract_table(doi: str, label: str, transcription_path: str, data_root: str,
                   material_description: str, measurement_purpose: str, baseline_material,
                   extraction_method: str = ExtractionMethod.OCR_EXTRACTED_TABLE.value,
                   overwrite: bool = False, curve_label: str = None) -> dict:
    paper_id = paper_id_from_doi(doi)
    paper_dir = Path(data_root) / "papers" / paper_id
    manifest_path = paper_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"{manifest_path} doesn't exist -- run ingest_paper.py first.")

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    region = _find_region(manifest, label)
    if region["kind"] not in ("table", "figure"):
        raise ValueError(f"{label!r} is a {region['kind']}, not a table or figure -- wrong tool for this region.")
    # A "figure" region ending up here (not extract_spectrum.py) means its
    # content is fundamentally discrete/tabular, not a continuous trace worth
    # pixel-digitizing -- e.g. this project's own Figure 7: sparse kinetics
    # data (4-7 marker points per curve, straight lines drawn between them
    # for visualization only). Forcing that through the curve digitizer would
    # fabricate false precision between real measurement points, so it's
    # transcribed as a table instead. curve_label follows extract_spectrum.py's
    # same multi-series-per-region pattern (see its module docstring) so
    # e.g. Figure 7's 11 curves across 2 panels get distinct, non-colliding
    # record_ids/filenames instead of overwriting each other.

    citation_meta = manifest.get("citation_metadata")
    if not citation_meta:
        raise RuntimeError(
            f"No citation_metadata in {manifest_path} -- run fetch_metadata.py before extract_table.py "
            f"so this table record can carry a real citation, not just a bare DOI."
        )
    if not citation_meta.get("metadata_reviewed"):
        raise RuntimeError(
            f"citation_metadata exists but hasn't been marked reviewed yet -- run "
            f"'python fetch_metadata.py --doi {doi} --mark-reviewed' first (after checking it), "
            f"so a table record is never built on top of unverified citation data."
        )

    tables_dir = paper_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    label_slug = label.replace(" ", "_").replace(".", "")
    curve_slug = None
    if curve_label:
        curve_slug = "".join(c if c.isalnum() else "_" for c in curve_label).strip("_")
        while "__" in curve_slug:
            curve_slug = curve_slug.replace("__", "_")
    safe_label = f"{label_slug}_{curve_slug}" if curve_slug else label_slug
    out_path = tables_dir / f"{safe_label}.json"
    if out_path.exists() and not overwrite:
        raise FileExistsError(f"{out_path} already exists -- pass --overwrite to redo it.")

    with open(transcription_path, "r", encoding="utf-8") as f:
        transcription = json.load(f)
    if "columns" not in transcription or "rows" not in transcription:
        raise ValueError(f"{transcription_path} must have top-level 'columns' and 'rows' keys.")

    baseline = None if (isinstance(baseline_material, str) and baseline_material.strip().lower() == "null") else baseline_material
    source_location = f"{region['label']} ({curve_label})" if curve_label else region["label"]

    record = {
        "record_id": f"{paper_id}_{safe_label.lower()}",
        "doi": manifest["doi"],
        "source_location": source_location,
        "license": manifest["license"],
        "citation": citation_meta["citation"],
        "material_description": material_description,
        "baseline_material": baseline,
        "measurement_purpose": measurement_purpose,
        "table_title": region["caption"],
        "columns": transcription["columns"],
        "rows": transcription["rows"],
        "source_type": SourceType.DIGITIZED_FIGURE.value,
        "extraction_method": extraction_method,
        "review_status": ReviewStatus.UNREVIEWED.value,
        "curator": "SpectraScribe (extract_table.py, transcription human/LLM-provided)",
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "peer_reviewed": True,   # ACS Omega is peer-reviewed; not auto-detected, stated by the pipeline's current single-source assumption
        "open_access": True,     # SOURCE_POLICY.md requires this to already be true before ingestion happens at all
        "cross_validated": False,
    }
    record["confidence_score"], record["confidence_breakdown"] = compute_confidence_score(record)

    problems = validate_table_record(record)
    if problems:
        raise ValueError(f"Transcribed table failed validation:\n  " + "\n  ".join(problems))

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2)

    return record


def mark_reviewed(doi: str, label: str, data_root: str, curve_label: str = None) -> None:
    paper_id = paper_id_from_doi(doi)
    label_slug = label.replace(" ", "_").replace(".", "")
    curve_slug = None
    if curve_label:
        curve_slug = "".join(c if c.isalnum() else "_" for c in curve_label).strip("_")
        while "__" in curve_slug:
            curve_slug = curve_slug.replace("__", "_")
    safe_label = f"{label_slug}_{curve_slug}" if curve_slug else label_slug
    out_path = Path(data_root) / "papers" / paper_id / "tables" / f"{safe_label}.json"
    if not out_path.exists():
        raise FileNotFoundError(f"{out_path} doesn't exist -- run extract_table.py first.")
    with open(out_path, "r", encoding="utf-8") as f:
        record = json.load(f)
    record["review_status"] = ReviewStatus.SELF_REVIEWED.value
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Phase 2: assemble a validated table record from a transcription.")
    parser.add_argument("--doi", required=True)
    parser.add_argument("--label", required=True, help='e.g. "Table S1" -- must match a table region in the manifest.')
    parser.add_argument("--transcription", help="Path to a {columns, rows} JSON transcription file.")
    parser.add_argument("--data-root", default="../data")
    parser.add_argument("--material-description", help="What sample/material this table's data was measured on.")
    parser.add_argument("--measurement-purpose", help="Why this measurement was made, in one sentence.")
    parser.add_argument("--baseline-material", help="What this table is compared against, or the literal string 'null' if there isn't one.")
    parser.add_argument("--extraction-method", default=ExtractionMethod.OCR_EXTRACTED_TABLE.value,
                         choices=[e.value for e in ExtractionMethod])
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--mark-reviewed", action="store_true")
    parser.add_argument("--curve-label", help='Which series/panel this run is for, e.g. "TiO2/Cu2O 60 min" -- '
                                               'required when one figure/table region holds more than one '
                                               'independent series. Appended to record_id/filenames/source_location '
                                               'so series from the same region never collide.')
    args = parser.parse_args()

    if args.mark_reviewed:
        mark_reviewed(args.doi, args.label, args.data_root, curve_label=args.curve_label)
        print(f"Marked {args.label!r} as self-reviewed.")
        return

    if not args.transcription:
        parser.error("--transcription is required unless --mark-reviewed is passed")
    if not args.material_description or not args.measurement_purpose or not args.baseline_material:
        parser.error("--material-description/--measurement-purpose/--baseline-material are required unless --mark-reviewed is passed")

    try:
        record = extract_table(args.doi, args.label, args.transcription, args.data_root,
                                args.material_description, args.measurement_purpose, args.baseline_material,
                                args.extraction_method, args.overwrite, curve_label=args.curve_label)
    except (FileExistsError, FileNotFoundError, ValueError, RuntimeError) as e:
        print(f"[ERROR] {e}")
        raise SystemExit(1)

    print(f"Wrote {record['record_id']}: {len(record['columns'])} columns x {len(record['rows'])} rows")
    print(f"confidence_score: {record['confidence_score']:.2f} {record['confidence_breakdown']}")
    print(f"[UNREVIEWED] -- check the transcription against data/papers/*/crops/*{args.label.replace(' ', '_')}*.png, "
          f"then run with --mark-reviewed")


if __name__ == "__main__":
    main()
