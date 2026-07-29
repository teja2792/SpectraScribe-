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
                   extraction_method: str = ExtractionMethod.OCR_EXTRACTED_TABLE.value,
                   overwrite: bool = False) -> dict:
    paper_id = paper_id_from_doi(doi)
    paper_dir = Path(data_root) / "papers" / paper_id
    manifest_path = paper_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"{manifest_path} doesn't exist -- run ingest_paper.py first.")

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    region = _find_region(manifest, label)
    if region["kind"] != "table":
        raise ValueError(f"{label!r} is a {region['kind']}, not a table -- wrong tool for this region.")

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
    safe_label = label.replace(" ", "_").replace(".", "")
    out_path = tables_dir / f"{safe_label}.json"
    if out_path.exists() and not overwrite:
        raise FileExistsError(f"{out_path} already exists -- pass --overwrite to redo it.")

    with open(transcription_path, "r", encoding="utf-8") as f:
        transcription = json.load(f)
    if "columns" not in transcription or "rows" not in transcription:
        raise ValueError(f"{transcription_path} must have top-level 'columns' and 'rows' keys.")

    record = {
        "record_id": f"{paper_id}_{safe_label.lower()}",
        "doi": manifest["doi"],
        "source_location": region["label"],
        "license": manifest["license"],
        "citation": citation_meta["citation"],
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


def mark_reviewed(doi: str, label: str, data_root: str) -> None:
    paper_id = paper_id_from_doi(doi)
    safe_label = label.replace(" ", "_").replace(".", "")
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
    parser.add_argument("--extraction-method", default=ExtractionMethod.OCR_EXTRACTED_TABLE.value,
                         choices=[e.value for e in ExtractionMethod])
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--mark-reviewed", action="store_true")
    args = parser.parse_args()

    if args.mark_reviewed:
        mark_reviewed(args.doi, args.label, args.data_root)
        print(f"Marked {args.label!r} as self-reviewed.")
        return

    if not args.transcription:
        parser.error("--transcription is required unless --mark-reviewed is passed")

    try:
        record = extract_table(args.doi, args.label, args.transcription, args.data_root,
                                args.extraction_method, args.overwrite)
    except (FileExistsError, FileNotFoundError, ValueError, RuntimeError) as e:
        print(f"[ERROR] {e}")
        raise SystemExit(1)

    print(f"Wrote {record['record_id']}: {len(record['columns'])} columns x {len(record['rows'])} rows")
    print(f"confidence_score: {record['confidence_score']:.2f} {record['confidence_breakdown']}")
    print(f"[UNREVIEWED] -- check the transcription against data/papers/*/crops/*{args.label.replace(' ', '_')}*.png, "
          f"then run with --mark-reviewed")


if __name__ == "__main__":
    main()
