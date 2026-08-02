"""One-off script: Figure S10 (Tauc plots for TiO2 and TiO2/Cu2O) is a
2-panel figure split into crops/_panels/Figure_S10{a,b}.png. Both panels'
automatic border-detection locked onto the ROTATED y-axis TITLE TEXT
block (e.g. "(alpha*hv)^1/2 (eV/cm)^1/2"), not the real axis line -- the
title text sits close enough to the left margin and is tall enough
(spans nearly the full plot height) to fool the tallest-column heuristic.
The TRUE axis line was found further right in both panels by widening the
column search past the title-text block (see chat) and confirmed against
the visible printed number labels. Real y-tick/x-tick pixel positions
were then found by scanning immediately left of (y) / below (x) that
corrected axis line.

Each plot has a SOLID curve (the actual Tauc-transformed absorption data)
and a DASHED straight line (the linear extrapolation the paper's authors
drew to read off the bandgap x-intercept) touching it in one place. Only
the solid curve is real data -- the dashed line is an annotation. Checked
the overlay for dashed-line contamination near the tangent point before
finalizing (see chat).
"""
import json, sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, "/sessions/elegant-happy-euler/mnt/CatalysisAI/SpectraScribe-/src")
from extract_spectrum import (_isolate_curve_component, _linear_calibration, write_overlay,
                               CURVE_DARK_THRESHOLD, BORDER_SEARCH_MARGIN, NEAR_VERTICAL_SPAN_PX,
                               MIN_CURVE_SEGMENT_PX)
from schema import Modality, SourceType, ExtractionMethod, ReviewStatus, compute_confidence_score, validate_record
import numpy as np
from scipy import ndimage
from PIL import Image

PAPER_DIR = Path("/sessions/elegant-happy-euler/mnt/CatalysisAI/SpectraScribe-/data/papers/acsomega_5c08505")
PANEL_DIR = PAPER_DIR / "crops" / "_panels"
SPECTRA_DIR = PAPER_DIR / "spectra"
DOI = "10.1021/acsomega.5c08505"

manifest = json.load(open(PAPER_DIR / "manifest.json"))
citation_meta = manifest["citation_metadata"]
assert citation_meta["metadata_reviewed"]


def digitize_panel(panel, border, x_ticks, y_ticks, x_first, x_last, y_first, y_last,
                    x_tick_step=None, y_tick_step=None):
    top, bottom, left, right = border
    path = str(PANEL_DIR / f"Figure_S10{panel}.png")
    gray = np.array(Image.open(path).convert("L"))
    x_slope, x_intercept, x_resid = _linear_calibration(x_ticks, x_first, x_last, x_tick_step)
    y_slope, y_intercept, y_resid = _linear_calibration(y_ticks, y_first, y_last, y_tick_step)
    ink = gray < CURVE_DARK_THRESHOLD
    # This figure's gridlines (gray=102) are darker than CURVE_DARK_THRESHOLD=180, so they get
    # matched as "curve ink" -- a full-plot-width horizontal gridline then passes
    # MIN_CURVE_SEGMENT_PX's width filter just like the real curve does, and _isolate_curve_component
    # unions it into curve_mask, corrupting every column with a second matched row far from the real
    # curve. Caught for real on panel (a): every one of 738 columns got flagged as "near-vertical"
    # because the real curve and the y=12 gridline were both present in the same column, ~600px apart.
    # Fixed by blanking a thin band at each known gridline row/col before isolation -- this costs a
    # few px of real curve at the handful of columns where it happens to cross a gridline, which is a
    # far smaller error than the contamination it prevents.
    # Only horizontal (y-tick) gridlines cause the full-column-spanning corruption described
    # above -- a vertical (x-tick) gridline only affects the one column it's in, which
    # _isolate_curve_component's per-column trace already handles fine (it's just part of
    # whatever's in that column), so it's left alone rather than fragmenting the curve into
    # many pieces (some too narrow to pass MIN_CURVE_SEGMENT_PX) for no real benefit.
    # Exclude every gridline EXCEPT the one at y=0 (last in the list, since y_ticks runs
    # high-to-low): a Tauc curve's baseline legitimately sits at (alpha*hv)^n = 0 for
    # sub-bandgap photon energies, so the real curve and the y=0 gridline are pixel-for-
    # pixel identical there -- excluding it blanked out the entire flat baseline on panel
    # (b) (only the near-bandgap rising edge survived). Nothing lost by leaving it in: a
    # real flat-zero baseline doesn't get corrupted by a coincident gridline the way a
    # sloped curve crossing a gridline briefly would.
    for gy in y_ticks[:-1]:
        r = int(round(gy))
        ink[max(0, r - 1):r + 2, :] = False
    interior = ink[top + BORDER_SEARCH_MARGIN: bottom - BORDER_SEARCH_MARGIN,
                    left + BORDER_SEARCH_MARGIN: right - BORDER_SEARCH_MARGIN]
    # Panel (b)'s steep near-bandgap rising edge crosses ~12 closely-spaced gridlines (every
    # 5 y-units, ~37.7px apart) in a narrow x-range. Because the curve is near-VERTICAL there,
    # each 3px gridline-exclusion band (above) doesn't just nick the curve -- it cuts clean
    # through it, splitting that stretch into many small pieces whose bounding box is only a
    # few px wide in x (span in _isolate_curve_component's sense), each well under
    # MIN_CURVE_SEGMENT_PX=45, so the plain component filter below discarded all of them:
    # panel (b) came back with only the flat baseline (601 pts, x maxing out at 2.56 eV
    # instead of the true ~2.8 eV) -- the entire steep transition was silently missing.
    # Confirmed the gridlines themselves are only ~1-2px dark (row scan: values
    # [254 254 254 254 223 185 234 249 254 254], single darkest pixel 185) so the exclusion
    # band is already near-minimal; narrowing it further wouldn't recover the fragments.
    # Fixed by bridging the gaps with a VERTICAL-ONLY dilation before connected-component
    # labeling -- reconnects pieces that were only ever severed by a gridline-exclusion band a
    # few px tall, without bridging unrelated features horizontally. The MIN_CURVE_SEGMENT_PX
    # filter is then applied to the bridged (dilated) components, but the final kept mask is
    # intersected back with the ORIGINAL undilated ink, so dilation only affects which pieces
    # get grouped together for the width test -- no fabricated pixels are added to the trace.
    bridge = np.zeros((5, 1), dtype=bool)
    bridge[:, 0] = True
    dilated = ndimage.binary_dilation(interior, structure=bridge)
    labeled, n = ndimage.label(dilated, structure=np.ones((3, 3)))
    curve_mask = np.zeros_like(interior, dtype=bool)
    for label_id in range(1, n + 1):
        comp = labeled == label_id
        cols = np.where(comp.any(axis=0))[0]
        if len(cols) == 0:
            continue
        if cols.max() - cols.min() >= MIN_CURVE_SEGMENT_PX:
            curve_mask |= comp & interior
    x_values, y_values, thick = [], [], []
    last_row = None
    n_corrected = 0
    for col in range(curve_mask.shape[1]):
        rows = np.where(curve_mask[:, col])[0]
        if len(rows) == 0:
            continue
        span = rows.max() - rows.min()
        if span > NEAR_VERTICAL_SPAN_PX:
            if last_row is None or abs(rows.min() - last_row) <= abs(rows.max() - last_row):
                pr = float(rows.min())
            else:
                pr = float(rows.max())
            n_corrected += 1
        else:
            pr = rows.mean()
        last_row = pr
        pr_px = pr + top + BORDER_SEARCH_MARGIN
        pc_px = col + left + BORDER_SEARCH_MARGIN
        x_values.append(x_slope * pc_px + x_intercept)
        y_values.append(y_slope * pr_px + y_intercept)
        thick.append(0.0 if span > NEAR_VERTICAL_SPAN_PX else span * abs(y_slope))
    err = float(y_resid + (np.mean(thick) / 2 if thick else 0))
    diag = {"n_x_ticks_used": len(x_ticks), "n_y_ticks_used": len(y_ticks),
            "x_calibration_residual_std": x_resid, "y_calibration_residual_std": y_resid,
            "n_curve_points": len(x_values), "n_near_vertical_columns_corrected": n_corrected,
            "border_box_px": {"top": top, "bottom": bottom, "left": left, "right": right}}
    d = {"x_values": x_values, "y_values": y_values, "digitization_error_estimate": err,
         "diagnostics": diag, "_curve_mask": curve_mask, "_border": (top, bottom, left, right)}
    overlay_path = str(SPECTRA_DIR / f"Figure_S10{panel}_overlay.png")
    write_overlay(path, d, overlay_path)
    return d, overlay_path, path


def build_record(panel, digitized, crop_path, overlay_path, material_formula, material_description,
                  measurement_purpose, x_axis, y_axis):
    safe_label = f"Figure_S10{panel}"
    record = {
        "record_id": f"acsomega_5c08505_{safe_label.lower()}",
        "material_formula": material_formula,
        "material_description": material_description,
        "baseline_material": None,
        "measurement_purpose": measurement_purpose,
        "modality": Modality.UV_VIS_TAUC.value,
        "x_axis": x_axis,
        "y_axis": y_axis,
        "x_values": digitized["x_values"],
        "y_values": digitized["y_values"],
        "digitization_error_estimate": digitized["digitization_error_estimate"],
        "source_type": SourceType.DIGITIZED_FIGURE.value,
        "extraction_method": ExtractionMethod.SEMI_AUTOMATED_DIGITIZER.value,
        "review_status": ReviewStatus.UNREVIEWED.value,
        "curator": "SpectraScribe (extract_figureS10_acsomega5c08505.py, one-off 2-panel split + algorithmic trace)",
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_name": citation_meta.get("journal") or "unknown",
        "source_url": f"https://doi.org/{DOI}",
        "doi": manifest["doi"],
        "source_location": f"Figure S10{panel}",
        "license": manifest["license"],
        "citation": citation_meta["citation"],
        "peer_reviewed": True,
        "open_access": True,
        "cross_validated": False,
        "notes": f"Digitized from manually-split panel crop {crop_path} (Figure S10 is a 2-panel figure; "
                 f"ingest_paper.py's region detector doesn't split multi-panel figures). Overlay at "
                 f"{overlay_path}. Diagnostics: {digitized['diagnostics']}. Only the SOLID Tauc curve was "
                 f"traced; the DASHED line in the source figure is the authors' linear bandgap-"
                 f"extrapolation annotation, not measured data, and was excluded.",
    }
    record["confidence_score"], record["confidence_breakdown"] = compute_confidence_score(record)
    problems = validate_record(record)
    if problems:
        raise ValueError(f"Figure S10{panel} failed validation: {problems}")
    out = SPECTRA_DIR / f"{safe_label}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2)
    print(f"Wrote {record['record_id']}: {len(record['x_values'])} points, "
          f"err={record['digitization_error_estimate']:.3f}, conf={record['confidence_score']:.2f}, "
          f"n_corrected={digitized['diagnostics']['n_near_vertical_columns_corrected']}")
    return record


# Panel a: TiO2, indirect-bandgap Tauc plot ((alpha*hv)^1/2 vs hv), paper reports 3.43 eV
xt_a = [506.0, 561.0, 615.5, 671.0, 725.5, 780.0, 834.5, 889.0, 944.0, 998.5, 1053.0, 1108.0, 1163.0]
yt_a = [430.5, 529.5, 629.0, 727.5, 827.5, 926.0, 1025.5]
d, ov, cp = digitize_panel("a", (396, 1030, 462, 1206), xt_a, yt_a,
                            x_first=2.0, x_last=5.0, y_first=12, y_last=0,
                            x_tick_step=0.25, y_tick_step=2)
build_record("a", d, cp, ov, "TiO2",
              "TiO2 mesoporous coating on glass substrate (same layer characterized structurally in Table S1/S2)",
              "Tauc plot analysis (indirect-allowed transition, n=1/2) of the TiO2 coating's UV-Vis "
              "absorption spectrum, to extract the optical bandgap by linear extrapolation of the "
              "absorption edge to zero -- paper reports 3.43 eV, consistent with anatase/mixed-phase TiO2",
              "hv (eV)", "(alpha*hv)^1/2 (eV/cm)^1/2")

# Panel b: TiO2/Cu2O composite, direct-bandgap Tauc plot ((alpha*hv)^2 vs hv), paper reports 2.49 eV
xt_b = [191.5, 242.0, 293.0, 344.0, 394.5, 445.0, 495.5, 546.0, 597.5, 648.5, 699.5, 750.0, 795.0, 852.0, 902.5]
yt_b = [425.5, 463.5, 501.0, 539.0, 576.0, 614.0, 651.0, 689.0, 727.0, 764.5, 803.0, 840.0, 878.0, 915.0, 953.0, 990.5, 1028.5]
d, ov, cp = digitize_panel("b", (396, 1032, 176, 964), xt_b, yt_b,
                            x_first=1.4, x_last=2.8, y_first=80, y_last=0,
                            x_tick_step=0.1, y_tick_step=5)
build_record("b", d, cp, ov, "TiO2/Cu2O",
              "TiO2/Cu2O composite coating (Cu2O deposited on the mesoporous TiO2 layer) -- reaction time "
              "for this specific sample not stated in the Tauc plot's own caption",
              "Tauc plot analysis (direct-allowed transition, n=2, appropriate for Cu2O) of the TiO2/Cu2O "
              "composite's UV-Vis absorption spectrum, to extract the composite's optical bandgap -- paper "
              "reports 2.49 eV, red-shifted from bare TiO2's 3.43 eV, consistent with Cu2O sensitization "
              "extending absorption into the visible range",
              "hv (eV)", "(alpha*hv)^2 (eV/cm)^2")

print("Done.")
