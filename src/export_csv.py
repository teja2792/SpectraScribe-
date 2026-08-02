"""Phase 6 (partial, SpectraScribe side): export every spectral and table
record across data/papers/*/ into flat CSV files a chemist can open in Excel
without touching Python or JSON.

Two record shapes exist in this repo (see schema.py):
  - Spectral records (data/papers/*/spectra/*.json): one curve, x_values/
    y_values arrays -- exported LONG (one row per data point) since a
    spectrum's whole point is the shape of the curve, not a single number.
  - Table records (data/papers/*/tables/*.json): one row_dict per
    measurement (e.g. Figure 7's kinetics: {"time_h": 1.0, "A_A0": 0.82}) --
    exported one CSV row per table row, with the row's own columns kept as
    separate CSV columns where a table is internally consistent, plus a
    catch-all row_data_json column so nothing is silently dropped for
    tables whose column sets vary from paper to paper.

Scope filtering (the actual reason this script exists, not an afterthought):
this project's explicit goal is intrinsic MATERIAL PROPERTIES (composition,
structure, size/shape/film properties, bandgap, ...), not reaction/process
PERFORMANCE metrics that vary with experimental conditions (e.g. Figure 7's
and Figure S14's photocatalytic dye-degradation kinetics vs time/light
source). Records that are genuine, reviewed data but out of that scope are
kept in the database (not deleted) and flagged out_of_scope=true -- see
schema.py's module docstring "Scope note". By DEFAULT this script excludes
those from the main export files (so "give me the material-property data"
doesn't require the user to know Figure 7 exists and filter it out by hand)
but always also writes a *_all.csv sibling with everything, out_of_scope
records included and clearly labeled, so nothing is ever actually hidden --
only kept out of the default view.

Every record's own review_status is preserved as a column rather than
silently filtered to reviewed-only, on purpose: "is this data trustworthy
enough for my use case" is a judgment call for whoever's consuming the CSV,
not something to make silently on their behalf. confidence_score is exported
for the same reason."""
import csv
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PAPERS_DIR = REPO_ROOT / "data" / "papers"
EXPORT_DIR = REPO_ROOT / "data" / "exports"

SPECTRA_COLUMNS = [
    "paper_id", "doi", "record_id", "source_location", "modality",
    "material_formula", "material_description", "baseline_material", "measurement_purpose",
    "x_axis", "y_axis", "x_value", "y_value",
    "digitization_error_estimate", "confidence_score", "review_status",
    "extraction_method", "source_type", "peer_reviewed", "open_access", "cross_validated",
    "out_of_scope", "out_of_scope_reason",
    "curator", "source_name", "source_url", "license", "citation", "retrieved_at_utc",
]

TABLE_COLUMNS = [
    "paper_id", "doi", "record_id", "source_location", "table_title",
    "material_description", "baseline_material", "measurement_purpose",
    "row_index", "row_data_json",
    "confidence_score", "review_status", "extraction_method", "source_type",
    "peer_reviewed", "open_access", "cross_validated",
    "out_of_scope", "out_of_scope_reason",
    "curator", "retrieved_at_utc", "license", "citation",
]


def _paper_id_from_path(json_path: Path) -> str:
    # .../data/papers/<paper_id>/spectra/Foo.json -> <paper_id>
    return json_path.parents[1].name


def _spectra_rows(record: dict, paper_id: str):
    x_values = record.get("x_values") or []
    y_values = record.get("y_values") or []
    base = {
        "paper_id": paper_id,
        "doi": record.get("doi"),
        "record_id": record.get("record_id"),
        "source_location": record.get("source_location"),
        "modality": record.get("modality"),
        "material_formula": record.get("material_formula"),
        "material_description": record.get("material_description"),
        "baseline_material": record.get("baseline_material"),
        "measurement_purpose": record.get("measurement_purpose"),
        "x_axis": record.get("x_axis"),
        "y_axis": record.get("y_axis"),
        "digitization_error_estimate": record.get("digitization_error_estimate"),
        "confidence_score": record.get("confidence_score"),
        "review_status": record.get("review_status"),
        "extraction_method": record.get("extraction_method"),
        "source_type": record.get("source_type"),
        "peer_reviewed": record.get("peer_reviewed"),
        "open_access": record.get("open_access"),
        "cross_validated": record.get("cross_validated"),
        "out_of_scope": bool(record.get("out_of_scope", False)),
        "out_of_scope_reason": record.get("out_of_scope_reason"),
        "curator": record.get("curator"),
        "source_name": record.get("source_name"),
        "source_url": record.get("source_url"),
        "license": record.get("license"),
        "citation": record.get("citation"),
        "retrieved_at_utc": record.get("retrieved_at_utc"),
    }
    for xv, yv in zip(x_values, y_values):
        row = dict(base)
        row["x_value"] = xv
        row["y_value"] = yv
        yield row


def _table_rows(record: dict, paper_id: str):
    base = {
        "paper_id": paper_id,
        "doi": record.get("doi"),
        "record_id": record.get("record_id"),
        "source_location": record.get("source_location"),
        "table_title": record.get("table_title"),
        "material_description": record.get("material_description"),
        "baseline_material": record.get("baseline_material"),
        "measurement_purpose": record.get("measurement_purpose"),
        "confidence_score": record.get("confidence_score"),
        "review_status": record.get("review_status"),
        "extraction_method": record.get("extraction_method"),
        "source_type": record.get("source_type"),
        "peer_reviewed": record.get("peer_reviewed"),
        "open_access": record.get("open_access"),
        "cross_validated": record.get("cross_validated"),
        "out_of_scope": bool(record.get("out_of_scope", False)),
        "out_of_scope_reason": record.get("out_of_scope_reason"),
        "curator": record.get("curator"),
        "retrieved_at_utc": record.get("retrieved_at_utc"),
        "license": record.get("license"),
        "citation": record.get("citation"),
    }
    for i, row_data in enumerate(record.get("rows", [])):
        row = dict(base)
        row["row_index"] = i
        row["row_data_json"] = json.dumps(row_data, sort_keys=True)
        yield row


def _write_csv(path: Path, columns: list, rows: list):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main():
    all_spectra_rows, all_table_rows = [], []
    n_spectra_records, n_table_records = 0, 0
    n_spectra_out_of_scope, n_table_out_of_scope = 0, 0

    for spectra_json in sorted(PAPERS_DIR.glob("*/spectra/*.json")):
        record = json.loads(spectra_json.read_text(encoding="utf-8"))
        paper_id = _paper_id_from_path(spectra_json)
        n_spectra_records += 1
        if record.get("out_of_scope"):
            n_spectra_out_of_scope += 1
        all_spectra_rows.extend(_spectra_rows(record, paper_id))

    for table_json in sorted(PAPERS_DIR.glob("*/tables/*.json")):
        record = json.loads(table_json.read_text(encoding="utf-8"))
        # Table_S1/Table_S2-style records are structural-property tables,
        # not row-per-measurement kinetics -- but they use the same
        # TABLE_RECORD_SCHEMA shape (rows: [{...}]), so no special-casing
        # needed here.
        paper_id = _paper_id_from_path(table_json)
        n_table_records += 1
        if record.get("out_of_scope"):
            n_table_out_of_scope += 1
        all_table_rows.extend(_table_rows(record, paper_id))

    in_scope_spectra = [r for r in all_spectra_rows if not r["out_of_scope"]]
    in_scope_tables = [r for r in all_table_rows if not r["out_of_scope"]]

    _write_csv(EXPORT_DIR / "spectra_material_properties.csv", SPECTRA_COLUMNS, in_scope_spectra)
    _write_csv(EXPORT_DIR / "spectra_all.csv", SPECTRA_COLUMNS, all_spectra_rows)
    _write_csv(EXPORT_DIR / "tables_material_properties.csv", TABLE_COLUMNS, in_scope_tables)
    _write_csv(EXPORT_DIR / "tables_all.csv", TABLE_COLUMNS, all_table_rows)

    print(f"Spectral records: {n_spectra_records} ({n_spectra_out_of_scope} out-of-scope) "
          f"-> {len(all_spectra_rows)} data points total, {len(in_scope_spectra)} in scope")
    print(f"Table records: {n_table_records} ({n_table_out_of_scope} out-of-scope) "
          f"-> {len(all_table_rows)} rows total, {len(in_scope_tables)} in scope")
    print(f"Wrote: {EXPORT_DIR / 'spectra_material_properties.csv'}")
    print(f"Wrote: {EXPORT_DIR / 'spectra_all.csv'}")
    print(f"Wrote: {EXPORT_DIR / 'tables_material_properties.csv'}")
    print(f"Wrote: {EXPORT_DIR / 'tables_all.csv'}")


if __name__ == "__main__":
    main()
