"""
extract_spectrum.py

Phase 3: semi-automated spectral curve digitization -- turn a cropped
plot image into (x_values, y_values) arrays, algorithmically, but never
without a human checking the result against the source image first (see
"Checkpoint" below). This is the one module in this project that does
real image processing rather than reading printed text/numbers, so it's
worth being explicit about what's automated and what isn't:

  AUTOMATED:
    - Finding the plot's axis border box (the black rectangular frame)
      by locating the rows/columns with the most dark pixels.
    - Finding tick mark pixel positions along the bottom and left edges
      (short perpendicular marks just outside the border).
    - Isolating "the curve" from other dark content inside the plot
      (peak-position labels like "557", "1101" printed directly over
      the curve in this project's real test figure) via connected-
      component analysis: the curve is the one component that spans
      most of the plot's width; text labels are small, disconnected
      blobs and are explicitly excluded, not accidentally traced into
      the data. This was found necessary, not assumed -- see
      _isolate_curve_component()'s docstring for how it was verified
      against a real figure with peak labels sitting on top of a peak.
    - Converting the isolated curve's pixel path to data coordinates,
      given axis calibration.

  NOT AUTOMATED -- requires a human-supplied answer, every time:
    - What data VALUE the first and last detected tick on each axis
      actually correspond to (--x-first-tick / --x-last-tick / etc.).
      Reading axis tick labels via OCR is exactly the kind of "looks
      right but is silently wrong" step this project avoids (see
      docs/SOURCE_POLICY.md's general stance and fetch_metadata.py's
      docstring for the same principle applied elsewhere) -- a misread
      digit here would silently mis-scale an entire curve. A human
      reads the two end labels off the actual figure once, instead.
    - Confirming the traced curve is actually right. Every run writes
      an overlay PNG (the original crop with the traced points drawn
      in red) specifically so this is a visual check, not a leap of
      faith -- see write_overlay(). A record is written with
      review_status="unreviewed" and MUST be promoted via
      --mark-reviewed (same checkpoint pattern as fetch_metadata.py and
      extract_table.py) after a human has actually looked at the
      overlay.

Known limitations, stated up front:
  - Single-curve figures only. A figure with multiple overlaid curves
    (e.g. Figure S11/S12's multi-timepoint absorption spectra) will have
    its curves picked up as one connected blob wherever they cross or
    run close together -- not handled here. Documented as a real
    follow-up, not silently mishandled.
  - Assumes ticks are evenly spaced between the first and last detected
    tick (true for every figure in this project's real test paper, and
    for the overwhelming majority of scientific plots, but not
    universally true of every possible axis -- e.g. a log axis would
    break this). No log-axis support yet.
  - Assumes a roughly white/light plot background and a dark
    (black/near-black) curve -- true for this project's real test
    figures, not guaranteed for a colored-curve or dark-theme plot.

Usage:
    python extract_spectrum.py --doi 10.xxxx/yyyy --label "Figure S2" \\
        --x-first-tick 100 --x-last-tick 1500 \\
        --y-first-tick 1000 --y-last-tick 0 \\
        --modality Raman --x-axis "Raman shift (cm-1)" --y-axis "Intensity (a.u.)" \\
        --material-formula "SiO2" --material-description "bare glass substrate (blank)" \\
        --measurement-purpose "confirm glass substrate's own Raman signal doesn't overlap sample peaks" \\
        --baseline-material null
    python extract_spectrum.py --doi 10.xxxx/yyyy --label "Figure S2" --mark-reviewed
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

from ingest_paper import paper_id_from_doi
from schema import ExtractionMethod, Modality, ReviewStatus, SourceType, compute_confidence_score, validate_record


DARK_THRESHOLD = 100        # grayscale value below which a pixel counts as "ink"
BORDER_SEARCH_MARGIN = 3    # px excluded on each side of a detected border line when scanning for ticks/curve
TICK_BAND_PX = 7            # how far outside the border to look for tick marks
TICK_MIN_DARK_FRACTION = 0.5  # fraction of the tick band that must be dark to count as a tick
MIN_CURVE_WIDTH_FRACTION = 0.5  # a connected component must span at least this fraction of the plot width to be "the curve", not a text label


def _find_border_box(dark: np.ndarray) -> tuple:
    """Locates the plot's rectangular axis border: the row/column with
    the most dark pixels in its half of the image, on each of the four
    sides. Returns (top, bottom, left, right) pixel coordinates. Assumes
    exactly one clean rectangular border, which is true of every figure
    in this project's real test data (standard OriginLab/SciDAVis-style
    plots) -- a figure without a drawn border box isn't handled."""
    h, w = dark.shape
    row_counts = dark.sum(axis=1)
    col_counts = dark.sum(axis=0)
    top = int(np.argmax(row_counts[: h // 2]))
    bottom = int(h // 2 + np.argmax(row_counts[h // 2 :]))
    left = int(np.argmax(col_counts[: w // 2]))
    right = int(w // 2 + np.argmax(col_counts[w // 2 :]))
    return top, bottom, left, right


def _find_ticks(dark: np.ndarray, top: int, bottom: int, left: int, right: int) -> tuple:
    """Finds tick-mark pixel centers along the bottom (x-axis) and left
    (y-axis) edges of the border box. A tick is a short run of dark
    pixels perpendicular to the axis, just outside the border line
    itself. Returns (x_tick_centers, y_tick_centers), both lists of
    pixel coordinates in ascending order (left-to-right for x,
    top-to-bottom for y)."""
    x_band = dark[bottom + BORDER_SEARCH_MARGIN : bottom + BORDER_SEARCH_MARGIN + TICK_BAND_PX, left:right]
    x_has_tick = x_band.sum(axis=0) >= TICK_MIN_DARK_FRACTION * TICK_BAND_PX
    x_tick_centers = [c + left for c in _group_consecutive(np.where(x_has_tick)[0])]

    y_band = dark[top:bottom, left - BORDER_SEARCH_MARGIN - TICK_BAND_PX : left - BORDER_SEARCH_MARGIN]
    y_has_tick = y_band.sum(axis=1) >= TICK_MIN_DARK_FRACTION * TICK_BAND_PX
    y_tick_centers = [c + top for c in _group_consecutive(np.where(y_has_tick)[0])]

    return x_tick_centers, y_tick_centers


def _group_consecutive(indices: np.ndarray, gap: int = 2) -> list:
    """Collapses runs of consecutive (or near-consecutive) pixel indices
    into single center points -- a tick mark is several pixels wide, and
    without this each tick would be counted many times over."""
    if len(indices) == 0:
        return []
    groups, cur = [], [indices[0]]
    for i in indices[1:]:
        if i - cur[-1] <= gap:
            cur.append(i)
        else:
            groups.append(cur)
            cur = [i]
    groups.append(cur)
    return [sum(g) / len(g) for g in groups]


def _isolate_curve_component(dark_interior: np.ndarray) -> np.ndarray:
    """Returns a boolean mask, same shape as dark_interior, containing
    ONLY the pixels belonging to the plotted curve -- not peak-position
    text labels ("557", "1101", ...) that this project's real test
    figure (Figure S2, a Raman spectrum) prints directly inside the plot
    area, sometimes touching or nearly touching the curve itself.

    Found necessary by checking, not assuming: a first version traced
    every dark pixel per column with no filtering, and produced a sudden
    spike wherever a label's digits added extra dark pixels above the
    real peak in that column. Fixed by connected-component labeling
    (scipy.ndimage.label) and keeping only the component(s) whose
    bounding box spans at least MIN_CURVE_WIDTH_FRACTION of the plot's
    width -- the real curve runs continuously across nearly the entire
    plot; a text label is a small, narrow, disconnected blob nowhere
    near that wide. Multiple curve segments that got broken into
    separate components (e.g. where the line dips below the DARK_THRESHOLD
    at a low-intensity noisy stretch) are all kept as long as each
    individually clears the width bar -- if the real curve is broken
    into several sub-threshold-width pieces, none dominant, that's a
    real failure mode this component-width heuristic won't catch,
    flagged here rather than pretending it's handled."""
    labeled, n = ndimage.label(dark_interior, structure=np.ones((3, 3)))
    width = dark_interior.shape[1]
    keep = np.zeros_like(dark_interior, dtype=bool)
    for label_id in range(1, n + 1):
        cols = np.where((labeled == label_id).any(axis=0))[0]
        if len(cols) == 0:
            continue
        span = cols.max() - cols.min()
        if span >= MIN_CURVE_WIDTH_FRACTION * width:
            keep |= labeled == label_id
    return keep


def _linear_calibration(pixel_positions: list, first_value: float, last_value: float) -> tuple:
    """Fits pixel -> data-value as a straight line through the first and
    last detected tick, using EVERY detected tick's assumed value
    (linearly interpolated between first_value and last_value, assuming
    even spacing -- true for all real figures this project has hit) as
    a least-squares check, not just the two endpoints -- this is what
    lets a residual/error estimate mean something: if the ticks aren't
    actually evenly spaced (a bad assumption for this figure), the fit
    residual will be large and visible in digitization_error_estimate,
    not silently swallowed by only ever using 2 points to define a line
    exactly. Returns (slope, intercept, residual_std)."""
    n = len(pixel_positions)
    assumed_values = np.linspace(first_value, last_value, n)
    slope, intercept = np.polyfit(pixel_positions, assumed_values, 1)
    predicted = slope * np.array(pixel_positions) + intercept
    residual_std = float(np.std(assumed_values - predicted))
    return float(slope), float(intercept), residual_std


def digitize(crop_path: str, x_first_tick: float, x_last_tick: float,
             y_first_tick: float, y_last_tick: float) -> dict:
    """Runs the full pipeline against one crop image. Returns a dict with
    x_values, y_values, digitization_error_estimate, and diagnostic info
    (tick counts found, calibration residuals) for the caller to inspect
    before deciding whether to trust the result at all."""
    img = Image.open(crop_path).convert("L")
    arr = np.array(img)
    dark = arr < DARK_THRESHOLD

    top, bottom, left, right = _find_border_box(dark)
    x_ticks, y_ticks = _find_ticks(dark, top, bottom, left, right)
    if len(x_ticks) < 2 or len(y_ticks) < 2:
        raise ValueError(
            f"Found only {len(x_ticks)} x-ticks and {len(y_ticks)} y-ticks -- need at least 2 per "
            f"axis to calibrate. This figure's tick marks may not match the expected style (short "
            f"marks just outside a rectangular border); check the crop manually."
        )

    x_slope, x_intercept, x_resid = _linear_calibration(x_ticks, x_first_tick, x_last_tick)
    y_slope, y_intercept, y_resid = _linear_calibration(y_ticks, y_first_tick, y_last_tick)

    interior = dark[top + BORDER_SEARCH_MARGIN : bottom - BORDER_SEARCH_MARGIN,
                     left + BORDER_SEARCH_MARGIN : right - BORDER_SEARCH_MARGIN]
    curve_mask = _isolate_curve_component(interior)

    x_values, y_values, col_thicknesses = [], [], []
    for col in range(curve_mask.shape[1]):
        rows_lit = np.where(curve_mask[:, col])[0]
        if len(rows_lit) == 0:
            continue
        row_centroid = rows_lit.mean()
        pixel_row = row_centroid + top + BORDER_SEARCH_MARGIN
        pixel_col = col + left + BORDER_SEARCH_MARGIN
        x_values.append(x_slope * pixel_col + x_intercept)
        y_values.append(y_slope * pixel_row + y_intercept)
        col_thicknesses.append((rows_lit.max() - rows_lit.min()) * abs(y_slope))

    if not x_values:
        raise ValueError("No curve pixels found after isolating the largest connected component -- check DARK_THRESHOLD and the crop image.")

    # digitization_error_estimate combines two independent sources: how
    # well the assumed-evenly-spaced ticks actually fit a line
    # (calibration uncertainty), and how thick the traced line itself is
    # in data units (a thick/noisy line has real ambiguity in "where
    # exactly is the curve," a hairline doesn't) -- reported as one
    # number, but the breakdown is kept in the manifest so it's
    # auditable, not a black box.
    error_estimate = float(y_resid + np.mean(col_thicknesses) / 2)

    return {
        "x_values": x_values,
        "y_values": y_values,
        "digitization_error_estimate": error_estimate,
        "diagnostics": {
            "n_x_ticks_found": len(x_ticks),
            "n_y_ticks_found": len(y_ticks),
            "x_calibration_residual_std": x_resid,
            "y_calibration_residual_std": y_resid,
            "n_curve_points": len(x_values),
            "border_box_px": {"top": top, "bottom": bottom, "left": left, "right": right},
        },
        "_curve_mask": curve_mask,
        "_border": (top, bottom, left, right),
    }


def write_overlay(crop_path: str, digitized: dict, out_path: str) -> None:
    """Saves the original crop with the traced curve drawn on top in red
    -- the actual human checkpoint. Nobody should trust a digitized
    curve without looking at this file at least once."""
    img = Image.open(crop_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    mask = digitized["_curve_mask"]
    top, bottom, left, right = digitized["_border"]
    ys, xs = np.where(mask)
    for x, y in zip(xs, ys):
        px = x + left + BORDER_SEARCH_MARGIN
        py = y + top + BORDER_SEARCH_MARGIN
        draw.point((px, py), fill=(255, 0, 0))
    img.save(out_path)


def main():
    parser = argparse.ArgumentParser(description="Phase 3: digitize a spectral curve from a cropped figure.")
    parser.add_argument("--doi", required=True)
    parser.add_argument("--label", required=True, help='e.g. "Figure S2" -- must match a figure region in the manifest.')
    parser.add_argument("--data-root", default="../data")
    parser.add_argument("--mark-reviewed", action="store_true")
    parser.add_argument("--overwrite", action="store_true")

    parser.add_argument("--x-first-tick", type=float, help="Data value of the leftmost detected x-tick (read off the actual figure).")
    parser.add_argument("--x-last-tick", type=float, help="Data value of the rightmost detected x-tick.")
    parser.add_argument("--y-first-tick", type=float, help="Data value of the topmost detected y-tick.")
    parser.add_argument("--y-last-tick", type=float, help="Data value of the bottommost detected y-tick.")
    parser.add_argument("--modality", choices=[m.value for m in Modality])
    parser.add_argument("--x-axis", help='e.g. "Raman shift (cm-1)"')
    parser.add_argument("--y-axis", help='e.g. "Intensity (a.u.)"')
    parser.add_argument("--material-formula")
    parser.add_argument("--material-description", help="Process-specific sample description, e.g. 'TiO2/Cu2O coating, 60 min reaction time'.")
    parser.add_argument("--measurement-purpose", help="Why this measurement was made, in one sentence.")
    parser.add_argument("--baseline-material", help="What this spectrum is compared against, or the literal string 'null' if there isn't one.")
    parser.add_argument("--extraction-method", default=ExtractionMethod.SEMI_AUTOMATED_DIGITIZER.value,
                         choices=[e.value for e in ExtractionMethod])
    args = parser.parse_args()

    paper_id = paper_id_from_doi(args.doi)
    paper_dir = Path(args.data_root) / "papers" / paper_id
    manifest_path = paper_dir / "manifest.json"
    safe_label = args.label.replace(" ", "_").replace(".", "")

    if args.mark_reviewed:
        rec_path = paper_dir / "spectra" / f"{safe_label}.json"
        with open(rec_path, "r", encoding="utf-8") as f:
            record = json.load(f)
        record["review_status"] = ReviewStatus.SELF_REVIEWED.value
        with open(rec_path, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2)
        print(f"Marked {args.label!r} as self-reviewed.")
        return

    required = [args.x_first_tick, args.x_last_tick, args.y_first_tick, args.y_last_tick,
                args.modality, args.x_axis, args.y_axis, args.material_formula,
                args.material_description, args.measurement_purpose, args.baseline_material]
    if any(v is None for v in required):
        parser.error("all of --x-first-tick/--x-last-tick/--y-first-tick/--y-last-tick/--modality/--x-axis/"
                     "--y-axis/--material-formula/--material-description/--measurement-purpose/"
                     "--baseline-material are required unless --mark-reviewed is passed")

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    region = next((r for r in manifest["regions"] if r["label"].lower().replace(" ", "") == args.label.lower().replace(" ", "")), None)
    if region is None:
        raise SystemExit(f"[ERROR] No region labeled {args.label!r} in manifest.")
    if region["kind"] != "figure":
        raise SystemExit(f"[ERROR] {args.label!r} is a {region['kind']}, not a figure.")

    citation_meta = manifest.get("citation_metadata")
    if not citation_meta or not citation_meta.get("metadata_reviewed"):
        raise SystemExit(f"[ERROR] citation_metadata missing or not yet reviewed for this paper -- run fetch_metadata.py (and --mark-reviewed) first.")

    spectra_dir = paper_dir / "spectra"
    spectra_dir.mkdir(parents=True, exist_ok=True)
    out_path = spectra_dir / f"{safe_label}.json"
    overlay_path = spectra_dir / f"{safe_label}_overlay.png"
    if out_path.exists() and not args.overwrite:
        raise SystemExit(f"[ERROR] {out_path} already exists -- pass --overwrite to redo it.")

    crop_path = region["crop_path"]
    try:
        digitized = digitize(crop_path, args.x_first_tick, args.x_last_tick, args.y_first_tick, args.y_last_tick)
    except ValueError as e:
        raise SystemExit(f"[ERROR] {e}")

    write_overlay(crop_path, digitized, str(overlay_path))

    baseline = None if args.baseline_material.strip().lower() == "null" else args.baseline_material

    record = {
        "record_id": f"{paper_id}_{safe_label.lower()}",
        "material_formula": args.material_formula,
        "material_description": args.material_description,
        "baseline_material": baseline,
        "measurement_purpose": args.measurement_purpose,
        "modality": args.modality,
        "x_axis": args.x_axis,
        "y_axis": args.y_axis,
        "x_values": digitized["x_values"],
        "y_values": digitized["y_values"],
        "digitization_error_estimate": digitized["digitization_error_estimate"],
        "source_type": SourceType.DIGITIZED_FIGURE.value,
        "extraction_method": args.extraction_method,
        "review_status": ReviewStatus.UNREVIEWED.value,
        "curator": "SpectraScribe (extract_spectrum.py, algorithmic trace + human calibration)",
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_name": citation_meta.get("journal") or "unknown",
        "source_url": f"https://doi.org/{args.doi}",
        "doi": manifest["doi"],
        "source_location": region["label"],
        "license": manifest["license"],
        "citation": citation_meta["citation"],
        "peer_reviewed": True,
        "open_access": True,
        "cross_validated": False,
        "notes": f"Digitized algorithmically from {crop_path}; overlay at {overlay_path}. "
                 f"Diagnostics: {digitized['diagnostics']}",
    }
    record["confidence_score"], record["confidence_breakdown"] = compute_confidence_score(record)

    problems = validate_record(record)
    if problems:
        raise SystemExit("[ERROR] Digitized record failed validation:\n  " + "\n  ".join(problems))

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2)

    print(f"Wrote {record['record_id']}: {len(record['x_values'])} points")
    print(f"Diagnostics: {digitized['diagnostics']}")
    print(f"digitization_error_estimate: {record['digitization_error_estimate']:.2f} {record['y_axis']} units")
    print(f"confidence_score: {record['confidence_score']:.2f}")
    print(f"[UNREVIEWED] -- open {overlay_path} and compare the red trace against the real curve, "
          f"then run with --mark-reviewed")


if __name__ == "__main__":
    main()
