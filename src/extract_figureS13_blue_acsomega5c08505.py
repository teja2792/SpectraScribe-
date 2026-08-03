"""One-off script: Figure S13's two TiO2/Cu2O (60 min) transmittance curves
("before" and "after 4h immersion in dye solution") are drawn in the exact
SAME blue RGB (0,0,255), distinguished only by line STYLE (solid vs dashed),
not color -- extract_spectrum.py's curve_color isolation (built for Figure
S12's same-hue-different-shade case) can't tell them apart on color alone.

Both curves are real data here (unlike Figure S10's dashed bandgap-
extrapolation line, which was an annotation to be excluded) -- the dashed
line is the "after immersion" measurement, not a construction aid.

Approach: per pixel column, find contiguous blue-ink runs (row ranges).
- 0 runs: no data at this x for either curve (shouldn't happen inside the
  curves' shared range; both curves span the full 350-1100 nm x-axis).
- 1 run: this is a column where the dash is "off" -- only the solid curve's
  ink is visible.  Assign it to the solid curve; the dashed curve gets no
  point at this x (a real, honest gap from the dash pattern, not fabricated
  through).
- >=2 runs: dash is "on" here, both curves visible and vertically separated.
  The dashed curve sits ABOVE the solid curve everywhere the two are
  distinguishable (confirmed by inspecting the source image: "after
  immersion" reads higher transmittance than "before" from ~350-950 nm,
  converging above ~950 nm) -- so the topmost run is dashed, the bottommost
  is solid.
Verified by a dedicated two-color overlay (green=solid, magenta=dashed)
checked against the source image before trusting this (see chat)."""
import json, sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, "/sessions/elegant-happy-euler/mnt/CatalysisAI/SpectraScribe-/src")
from extract_spectrum import (_find_border_box, _find_ticks, _linear_calibration, _color_distance_mask,
                               BORDER_DETECT_THRESHOLD, BORDER_SEARCH_MARGIN, NEAR_VERTICAL_SPAN_PX)
from schema import Modality, SourceType, ExtractionMethod, ReviewStatus, compute_confidence_score, validate_record
import numpy as np
from PIL import Image, ImageDraw

PAPER_DIR = Path("/sessions/elegant-happy-euler/mnt/CatalysisAI/SpectraScribe-/data/papers/acsomega_5c08505")
CROP_PATH = str(PAPER_DIR / "crops" / "si_page009_Figure_S13.png")
SPECTRA_DIR = PAPER_DIR / "spectra"
DOI = "10.1021/acsomega.5c08505"
LEGEND_BOX = (480, 540, 595, 725)  # x1,y1,x2,y2 -- covers all 4 legend swatch lines

manifest = json.load(open(PAPER_DIR / "manifest.json"))
citation_meta = manifest["citation_metadata"]
assert citation_meta["metadata_reviewed"]

gray = np.array(Image.open(CROP_PATH).convert("L"))
border_mask = gray < BORDER_DETECT_THRESHOLD
top, bottom, left, right = _find_border_box(border_mask)
x_ticks, y_ticks = _find_ticks(border_mask, top, bottom, left, right)
x_slope, x_intercept, x_resid = _linear_calibration(x_ticks, 350, 1100)
y_slope, y_intercept, y_resid = _linear_calibration(y_ticks, 100, 10, tick_step=10)

rgb = np.array(Image.open(CROP_PATH).convert("RGB"))
blue_mask = _color_distance_mask(rgb, (0, 0, 255), 60)
# Sub-pixel weight (how close each pixel actually is to pure blue, not just
# whether it passed the boolean tolerance) -- same fix as
# trace_curve_columns() in extract_spectrum.py, applied here by hand since
# this script's two-runs-per-column solid/dashed split doesn't fit that
# function's one-row-set-per-column shape. See that function's docstring
# for why a plain mean/midpoint jitters by a whole pixel on a thin line.
_diff = rgb.astype(np.float64) - np.array((0, 0, 255), dtype=np.float64)
blue_weight = np.clip(60 - np.sqrt((_diff ** 2).sum(axis=-1)), 0.0, None)
bx1, by1, bx2, by2 = LEGEND_BOX
blue_mask[by1:by2, bx1:bx2] = False
blue_weight[by1:by2, bx1:bx2] = 0.0
interior = blue_mask[top + BORDER_SEARCH_MARGIN: bottom - BORDER_SEARCH_MARGIN,
                      left + BORDER_SEARCH_MARGIN: right - BORDER_SEARCH_MARGIN]
weight_interior = blue_weight[top + BORDER_SEARCH_MARGIN: bottom - BORDER_SEARCH_MARGIN,
                               left + BORDER_SEARCH_MARGIN: right - BORDER_SEARCH_MARGIN]


def _weighted_centroid(lo, hi, col, w_interior):
    rows = np.arange(lo, hi + 1)
    w = w_interior[rows, col]
    return float(np.average(rows, weights=w)) if w.sum() > 0 else (lo + hi) / 2.0


solid_x, solid_y, solid_thick, solid_is_peak = [], [], [], []
dash_x, dash_y, dash_thick, dash_is_peak = [], [], [], []
last_solid_row, last_dash_row = None, None
n_solid_corrected, n_dash_corrected = 0, 0

for col in range(interior.shape[1]):
    rows = np.where(interior[:, col])[0]
    if len(rows) == 0:
        continue
    runs = []
    start = prev = rows[0]
    for r in rows[1:]:
        if r - prev > 3:
            runs.append((start, prev))
            start = r
        prev = r
    runs.append((start, prev))

    # solid = bottommost run (always present -- solid line is continuous)
    s_lo, s_hi = runs[-1]
    span = s_hi - s_lo
    is_peak = span > NEAR_VERTICAL_SPAN_PX
    if is_peak:
        pr = float(s_lo) if (last_solid_row is None or abs(s_lo - last_solid_row) <= abs(s_hi - last_solid_row)) else float(s_hi)
        n_solid_corrected += 1
    else:
        pr = _weighted_centroid(s_lo, s_hi, col, weight_interior)
    last_solid_row = pr
    pr_px = pr + top + BORDER_SEARCH_MARGIN
    pc_px = col + left + BORDER_SEARCH_MARGIN
    solid_x.append(x_slope * pc_px + x_intercept)
    solid_y.append(y_slope * pr_px + y_intercept)
    solid_thick.append(0.0 if is_peak else span * abs(y_slope))
    solid_is_peak.append(is_peak)

    # dashed = topmost run, only when a genuinely separate second run exists
    if len(runs) >= 2:
        d_lo, d_hi = runs[0]
        dspan = d_hi - d_lo
        d_is_peak = dspan > NEAR_VERTICAL_SPAN_PX
        if d_is_peak:
            pr = float(d_lo) if (last_dash_row is None or abs(d_lo - last_dash_row) <= abs(d_hi - last_dash_row)) else float(d_hi)
            n_dash_corrected += 1
        else:
            pr = _weighted_centroid(d_lo, d_hi, col, weight_interior)
        last_dash_row = pr
        pr_px = pr + top + BORDER_SEARCH_MARGIN
        dash_x.append(x_slope * pc_px + x_intercept)
        dash_y.append(y_slope * pr_px + y_intercept)
        dash_thick.append(0.0 if d_is_peak else dspan * abs(y_slope))
        dash_is_peak.append(d_is_peak)


def _smooth(y_values, is_peak_list, window=9):
    """Same light median de-noising as trace_curve_columns() -- see that
    function's docstring in extract_spectrum.py for why a small rolling
    median (not mean) is the right tool here, applied only to non-peak
    points. Returns (smoothed_y, jitter_std)."""
    y_arr = np.array(y_values)
    is_peak_arr = np.array(is_peak_list)
    if len(y_arr) < window or is_peak_arr.all():
        return y_values, 0.0
    half = window // 2
    padded = np.pad(y_arr, half, mode="edge")
    windows = np.lib.stride_tricks.sliding_window_view(padded, window)
    y_median = np.median(windows, axis=1)
    jitter_std = float(np.std((y_arr - y_median)[~is_peak_arr])) if (~is_peak_arr).any() else 0.0
    return np.where(is_peak_arr, y_arr, y_median).tolist(), jitter_std


solid_y, solid_jitter_std = _smooth(solid_y, solid_is_peak)
dash_y, dash_jitter_std = _smooth(dash_y, dash_is_peak)

# --- verification overlay: green=solid, magenta=dashed, drawn on the real crop ---
overlay_img = Image.open(CROP_PATH).convert("RGB")
draw = ImageDraw.Draw(overlay_img)
def draw_series(xs_data, ys_data, color):
    for xv, yv in zip(xs_data, ys_data):
        px = (xv - x_intercept) / x_slope
        py = (yv - y_intercept) / y_slope
        draw.ellipse((px - 1, py - 1, px + 1, py + 1), fill=color)
draw_series(solid_x, solid_y, (0, 200, 0))
draw_series(dash_x, dash_y, (230, 0, 230))
combined_overlay_path = str(SPECTRA_DIR / "Figure_S13_blue_both_overlay.png")
overlay_img.save(combined_overlay_path)
print(f"Wrote combined verification overlay: {combined_overlay_path}")
print(f"solid: {len(solid_x)} pts (n_near_vertical_corrected={n_solid_corrected}), "
      f"x=[{min(solid_x):.1f},{max(solid_x):.1f}], y=[{min(solid_y):.2f},{max(solid_y):.2f}]")
print(f"dashed: {len(dash_x)} pts (n_near_vertical_corrected={n_dash_corrected}), "
      f"x=[{min(dash_x):.1f},{max(dash_x):.1f}], y=[{min(dash_y):.2f},{max(dash_y):.2f}]")


def build_record(curve_label, xs_data, ys_data, thick, n_corrected, jitter_std, material_description,
                  measurement_purpose, baseline_material, notes_extra):
    safe_label = f"Figure_S13_{curve_label}"
    err = float(y_resid + (np.mean(thick) / 2 if thick else 0) + jitter_std)
    record = {
        "record_id": f"acsomega_5c08505_{safe_label.lower()}",
        "material_formula": "TiO2/Cu2O",
        "material_description": material_description,
        "baseline_material": baseline_material,
        "measurement_purpose": measurement_purpose,
        "modality": Modality.UV_VIS.value,
        "x_axis": "Wavelength (nm)",
        "y_axis": "Transmittance (%)",
        "x_values": xs_data,
        "y_values": ys_data,
        "digitization_error_estimate": err,
        "source_type": SourceType.DIGITIZED_FIGURE.value,
        "extraction_method": ExtractionMethod.SEMI_AUTOMATED_DIGITIZER.value,
        "review_status": "self-reviewed",
        "curator": "SpectraScribe (extract_figureS13_blue_acsomega5c08505.py, one-off same-color "
                   "solid-vs-dashed line-style separation)",
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_name": citation_meta.get("journal") or "unknown",
        "source_url": f"https://doi.org/{DOI}",
        "doi": manifest["doi"],
        "source_location": "Figure S13",
        "license": manifest["license"],
        "citation": citation_meta["citation"],
        "peer_reviewed": True,
        "open_access": True,
        "cross_validated": False,
        "notes": f"Digitized from {CROP_PATH}. This figure's two TiO2/Cu2O curves share the identical "
                 f"RGB (0,0,255) and are distinguished only by line style (solid vs dashed); separated by "
                 f"per-column run analysis of the blue mask -- solid = bottommost run in each column "
                 f"(always present), dashed = topmost run only in columns where the dash is 'on' "
                 f"(n={n_corrected} near-vertical columns corrected). Verified against a dedicated "
                 f"green(solid)/magenta(dashed) overlay at {combined_overlay_path} (write_overlay's "
                 f"built-in red trace would be indistinguishable from this figure's real red TiO2 curve). "
                 f"{notes_extra}",
    }
    record["confidence_score"], record["confidence_breakdown"] = compute_confidence_score(record)
    problems = validate_record(record)
    if problems:
        raise ValueError(f"{safe_label} failed validation: {problems}")
    out = SPECTRA_DIR / f"{safe_label}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2)
    print(f"Wrote {record['record_id']}: {len(record['x_values'])} points, "
          f"err={record['digitization_error_estimate']:.3f}, conf={record['confidence_score']:.2f}")
    return record


build_record(
    "tio2_cu2o_60min", solid_x, solid_y, solid_thick, n_solid_corrected, solid_jitter_std,
    "TiO2/Cu2O composite coating on glass, 60 min Cu2O deposition reaction time, BEFORE immersion in "
    "methyl orange dye solution",
    "UV-Vis transmittance of the as-prepared TiO2/Cu2O composite coating, the 'before' reference for the "
    "same sample's transmittance after 4h dye immersion (also in this figure)",
    "glass substrate and bare TiO2 coating, same figure",
    "Solid curve, continuous line style in the source figure -- traced as the bottommost blue run in "
    "every column (present at all 852 x-columns, no gaps)."
)

build_record(
    "tio2_cu2o_60min_after_4h_dye_immersion", dash_x, dash_y, dash_thick, n_dash_corrected, dash_jitter_std,
    "TiO2/Cu2O composite coating on glass, 60 min Cu2O deposition reaction time, AFTER 4h immersion in "
    "methyl orange dye solution (same physical sample as the 'before' curve in this same figure)",
    "UV-Vis transmittance of the same TiO2/Cu2O composite coating after 4h immersion in methyl orange dye "
    "solution, to assess whether dye adsorption onto the film changes its optical transmittance",
    "TiO2/Cu2O (60 min), same sample before dye immersion, same figure",
    "Dashed curve in the source figure -- only traced where the dash mark is actually drawn (the dash "
    "pattern's gaps are real, honest gaps in the output, not fabricated across); converges with the solid "
    "curve above ~950 nm where the two become visually indistinguishable."
)
