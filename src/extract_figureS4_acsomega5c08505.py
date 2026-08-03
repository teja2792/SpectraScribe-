"""One-off script: Figure S4 (XPS survey spectrum) has OPEN-style axes --
only left (y) and bottom (x) lines drawn, no top/right border box. The
general-purpose _find_border_box()/_find_ticks() in extract_spectrum.py
assume a full rectangular border and fail on this crop (found spurious
columns/rows instead of the real axis). Border and tick pixel positions
were determined manually via direct pixel inspection (see chat) instead:
y-axis ticks found just outside (to the left of) the vertical axis line at
column 877, x-axis ticks found just below the horizontal axis line at row
785 -- both confirmed against the visible printed tick labels.
This also exercises the NEAR_VERTICAL_SPAN_PX fix in extract_spectrum.py
(imported, not duplicated) since XPS core-level peaks are far sharper than
this project's other modalities: the Cu 2p peak's true height (~1,102,526
cps, confirmed by finding the single highest-intensity matched pixel
directly) was understated by ~30% (to ~773,000) under plain column-
averaging before that fix existed.
"""
import json, sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, "/sessions/elegant-happy-euler/mnt/CatalysisAI/SpectraScribe-/src")
from extract_spectrum import _isolate_curve_component, _linear_calibration, NEAR_VERTICAL_SPAN_PX
from schema import Modality, SourceType, ExtractionMethod, ReviewStatus, compute_confidence_score, validate_record
import numpy as np
from PIL import Image

PAPER_DIR = Path("/sessions/elegant-happy-euler/mnt/CatalysisAI/SpectraScribe-/data/papers/acsomega_5c08505")
SPECTRA_DIR = PAPER_DIR / "spectra"
DOI = "10.1021/acsomega.5c08505"
CROP = str(PAPER_DIR / "crops" / "si_page004_Figure_S4.png")

manifest = json.load(open(PAPER_DIR / "manifest.json"))
citation_meta = manifest["citation_metadata"]
assert citation_meta["metadata_reviewed"]
region = next(r for r in manifest["regions"] if r["label"] == "Figure S4")

rgb = np.array(Image.open(CROP).convert("RGB"))
r, g, b = rgb[..., 0].astype(int), rgb[..., 1].astype(int), rgb[..., 2].astype(int)
red_mask = (r > 180) & (g < 120) & (b < 120)
# Sub-pixel weight for the centroid below -- how much MORE red than
# green/blue each pixel is, not just whether it passed the (r>180)&
# (g<120)&(b<120) cutoff. Same purpose as trace_curve_columns()'s weight
# array in extract_spectrum.py (see that function's docstring); applied
# by hand here since this script predates that refactor and uses a
# different mask construction (RGB channel comparison, not
# _color_distance_mask) that doesn't produce the same weight signal.
red_weight = np.clip(r - np.maximum(g, b), 0, None).astype(np.float64)

x_ticks = [942.0, 1071.0, 1200.0, 1328.0, 1457.0, 1585.0, 1714.0]  # -> 1200,1000,800,600,400,200,0 eV
y_ticks = [111.0, 207.0, 303.0, 399.0, 495.0, 591.5, 688.0, 784.0]  # -> 1400000..0 cps, step 200000
x_slope, x_intercept, x_resid = _linear_calibration(x_ticks, 1200, 0, 200)
y_slope, y_intercept, y_resid = _linear_calibration(y_ticks, 1400000, 0, 200000)

curve_mask = _isolate_curve_component(red_mask)
x_values, y_values, thick, is_peak_list = [], [], [], []
last_row = None
n_corrected = 0
for col in range(curve_mask.shape[1]):
    rows = np.where(curve_mask[:, col])[0]
    if len(rows) == 0:
        continue
    span = rows.max() - rows.min()
    is_peak = span > NEAR_VERTICAL_SPAN_PX
    if is_peak:
        if last_row is None or abs(rows.min() - last_row) <= abs(rows.max() - last_row):
            pr = float(rows.min())
        else:
            pr = float(rows.max())
        n_corrected += 1
    else:
        w = red_weight[rows, col]
        pr = float(np.average(rows, weights=w)) if w.sum() > 0 else float(rows.mean())
    last_row = pr
    x_values.append(x_slope * col + x_intercept)
    y_values.append(y_slope * pr + y_intercept)
    thick.append(0.0 if is_peak else span * abs(y_slope))
    is_peak_list.append(is_peak)

# Light median de-noising (non-peak columns only) -- same technique as
# trace_curve_columns() in extract_spectrum.py; see that function's
# docstring for why. Negligible effect here in absolute terms (XPS
# intensities span hundreds of thousands of cps, so a few cps of pixel
# jitter is not the dominant error source the way it was for this
# project's thin, nearly-flat UV-Vis baseline curves), but applied for
# consistency rather than leaving one modality unfixed.
y_arr = np.array(y_values)
is_peak_arr = np.array(is_peak_list)
smooth_window = 9
if len(y_arr) >= smooth_window and not is_peak_arr.all():
    half = smooth_window // 2
    padded = np.pad(y_arr, half, mode="edge")
    windows = np.lib.stride_tricks.sliding_window_view(padded, smooth_window)
    y_median = np.median(windows, axis=1)
    jitter_std = float(np.std((y_arr - y_median)[~is_peak_arr])) if (~is_peak_arr).any() else 0.0
    y_values = np.where(is_peak_arr, y_arr, y_median).tolist()
else:
    jitter_std = 0.0

err = float(y_resid + (np.mean(thick) / 2 if thick else 0) + jitter_std)
diag = {"n_x_ticks_found": len(x_ticks), "n_y_ticks_found": len(y_ticks),
        "x_calibration_residual_std": x_resid, "y_calibration_residual_std": y_resid,
        "n_curve_points": len(x_values), "n_near_vertical_columns_corrected": n_corrected,
        "pixel_jitter_std_removed_by_smoothing": jitter_std,
        "border_note": "open-style axes (no top/right border) -- see script docstring"}

# overlay
img = Image.open(CROP).convert("RGB")
arr = np.array(img)
ys, xs = np.where(curve_mask)
arr[ys, xs] = [0, 255, 0]
overlay_path = str(SPECTRA_DIR / "Figure_S4_overlay.png")
Image.fromarray(arr).save(overlay_path)

record = {
    "record_id": "acsomega_5c08505_figure_s4",
    "material_formula": "TiO2/Cu2O",
    "material_description": "TiO2/Cu2O coating, 60 min reaction time, deposited on a silicon substrate (a separate witness sample specifically for XPS -- not the glass substrates used for the optical/photocatalytic measurements elsewhere in this paper)",
    "baseline_material": None,
    "measurement_purpose": "Survey scan to determine the sample's overall elemental composition (atomic % of each detected element) and confirm no Si 2p signal is detected (i.e. the coating is thick enough to fully attenuate the substrate signal)",
    "modality": Modality.XPS.value,
    "x_axis": "Binding energy (eV)",
    "y_axis": "Intensity (cps)",
    "x_values": x_values,
    "y_values": y_values,
    "digitization_error_estimate": err,
    "source_type": SourceType.DIGITIZED_FIGURE.value,
    "extraction_method": ExtractionMethod.SEMI_AUTOMATED_DIGITIZER.value,
    "review_status": ReviewStatus.UNREVIEWED.value,
    "curator": "SpectraScribe (extract_figureS4_acsomega5c08505.py, one-off open-axes handling + algorithmic trace)",
    "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
    "source_name": citation_meta.get("journal") or "unknown",
    "source_url": f"https://doi.org/{DOI}",
    "doi": manifest["doi"],
    "source_location": "Figure S4",
    "license": manifest["license"],
    "citation": citation_meta["citation"],
    "peer_reviewed": True,
    "open_access": True,
    "cross_validated": False,
    "notes": f"Digitized from {CROP}, a survey (wide-scan) XPS spectrum with OPEN-style axes (no drawn "
             f"top/right border -- see script docstring). Overlay at {overlay_path}. Diagnostics: {diag}. "
             f"Peak assignments per the source figure's own labels: Cu 2p (~932-933 eV, dominant peak), "
             f"Cu LMM Auger, O 1s, N 1s, C 1s, S 2p, Cu 3s, Cu 3p. Companion high-resolution Cu 2p/Auger "
             f"spectra are in Figure S5.",
}
record["confidence_score"], record["confidence_breakdown"] = compute_confidence_score(record)
problems = validate_record(record)
if problems:
    raise SystemExit(f"FAILED: {problems}")
out = SPECTRA_DIR / "Figure_S4.json"
with open(out, "w", encoding="utf-8") as f:
    json.dump(record, f, indent=2)
print(f"Wrote {record['record_id']}: {len(x_values)} points, err={err:.1f}, conf={record['confidence_score']:.2f}, n_corrected={n_corrected}")
