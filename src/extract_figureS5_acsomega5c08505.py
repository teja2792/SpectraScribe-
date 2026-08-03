"""One-off script: Figure S5 (a: Cu 2p high-res XPS, b: Cu L3M4,5M4,5 Auger)
is a 2-panel figure split into crops/_panels/Figure_S5{a,b}.png (whitest-
column-near-midpoint split, same technique as Figure 3). Both panels have
real rectangular borders + gridlines (unlike Figure S4's open axes), but
_find_ticks() under-detects: panel (a)'s automatic border-top (114) misses
the true top gridline (68, exactly at the '200000' label) and only finds 3
of the 4 real y-ticks. Y-tick pixel positions for panel (a) were confirmed
manually by scanning immediately left of the axis line and cross-checking
against a zoomed crop of the printed labels (see chat) -- 68/236/404/571.5
-> 200000/150000/100000/50000, evenly spaced, confirming these (not the
algorithm's 3-tick subset) are correct. Panel (b)'s ticks matched cleanly.
"""
import json, sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, "/sessions/elegant-happy-euler/mnt/CatalysisAI/SpectraScribe-/src")
from extract_spectrum import (_find_border_box, _isolate_curve_component, _linear_calibration,
                               write_overlay, BORDER_DETECT_THRESHOLD, CURVE_DARK_THRESHOLD,
                               BORDER_SEARCH_MARGIN, NEAR_VERTICAL_SPAN_PX, trace_curve_columns)
from schema import Modality, SourceType, ExtractionMethod, ReviewStatus, compute_confidence_score, validate_record
import numpy as np
from PIL import Image

PAPER_DIR = Path("/sessions/elegant-happy-euler/mnt/CatalysisAI/SpectraScribe-/data/papers/acsomega_5c08505")
PANEL_DIR = PAPER_DIR / "crops" / "_panels"
SPECTRA_DIR = PAPER_DIR / "spectra"
DOI = "10.1021/acsomega.5c08505"

manifest = json.load(open(PAPER_DIR / "manifest.json"))
citation_meta = manifest["citation_metadata"]
assert citation_meta["metadata_reviewed"]

MATERIAL_DESC = ("TiO2/Cu2O coating, 60 min reaction time, deposited on a silicon substrate "
                  "(same XPS witness sample as Figure S4)")


def digitize_panel(panel, x_ticks, y_ticks, x_first, x_last, y_first, y_last, x_tick_step=None, y_tick_step=None):
    path = str(PANEL_DIR / f"Figure_S5{panel}.png")
    gray = np.array(Image.open(path).convert("L"))
    border_mask = gray < BORDER_DETECT_THRESHOLD
    top, bottom, left, right = _find_border_box(border_mask)
    x_slope, x_intercept, x_resid = _linear_calibration(x_ticks, x_first, x_last, x_tick_step)
    y_slope, y_intercept, y_resid = _linear_calibration(y_ticks, y_first, y_last, y_tick_step)
    ink = gray < CURVE_DARK_THRESHOLD
    weight = np.clip(CURVE_DARK_THRESHOLD - gray.astype(np.float64), 0.0, None)
    interior = ink[top + BORDER_SEARCH_MARGIN: bottom - BORDER_SEARCH_MARGIN,
                    left + BORDER_SEARCH_MARGIN: right - BORDER_SEARCH_MARGIN]
    weight_interior = weight[top + BORDER_SEARCH_MARGIN: bottom - BORDER_SEARCH_MARGIN,
                              left + BORDER_SEARCH_MARGIN: right - BORDER_SEARCH_MARGIN]
    curve_mask = _isolate_curve_component(interior)
    # Shared tracer (near-vertical-peak fix, sub-pixel weighted centroid,
    # median de-noising) -- see trace_curve_columns()'s docstring in
    # extract_spectrum.py.
    trace = trace_curve_columns(curve_mask, weight_interior, top, left, x_slope, x_intercept,
                                 y_slope, y_intercept, y_resid, x_first, x_last)
    x_values, y_values = trace["x_values"], trace["y_values"]
    diag = {"n_x_ticks_used": len(x_ticks), "n_y_ticks_used": len(y_ticks),
            "x_calibration_residual_std": x_resid, "y_calibration_residual_std": y_resid,
            **trace["diagnostics"],
            "border_box_px": {"top": top, "bottom": bottom, "left": left, "right": right}}
    d = {"x_values": x_values, "y_values": y_values,
         "digitization_error_estimate": trace["digitization_error_estimate"],
         "diagnostics": diag, "_curve_mask": curve_mask, "_border": (top, bottom, left, right)}
    overlay_path = str(SPECTRA_DIR / f"Figure_S5{panel}_overlay.png")
    write_overlay(path, d, overlay_path)
    return d, overlay_path, path


def build_record(panel, digitized, crop_path, overlay_path, modality, x_axis, y_axis,
                  measurement_purpose, baseline_material):
    safe_label = f"Figure_S5{panel}"
    record = {
        "record_id": f"acsomega_5c08505_{safe_label.lower()}",
        "material_formula": "TiO2/Cu2O",
        "material_description": MATERIAL_DESC,
        "baseline_material": baseline_material,
        "measurement_purpose": measurement_purpose,
        "modality": modality,
        "x_axis": x_axis,
        "y_axis": y_axis,
        "x_values": digitized["x_values"],
        "y_values": digitized["y_values"],
        "digitization_error_estimate": digitized["digitization_error_estimate"],
        "source_type": SourceType.DIGITIZED_FIGURE.value,
        "extraction_method": ExtractionMethod.SEMI_AUTOMATED_DIGITIZER.value,
        "review_status": ReviewStatus.UNREVIEWED.value,
        "curator": "SpectraScribe (extract_figureS5_acsomega5c08505.py, one-off 2-panel split + algorithmic trace)",
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_name": citation_meta.get("journal") or "unknown",
        "source_url": f"https://doi.org/{DOI}",
        "doi": manifest["doi"],
        "source_location": f"Figure S5{panel}",
        "license": manifest["license"],
        "citation": citation_meta["citation"],
        "peer_reviewed": True,
        "open_access": True,
        "cross_validated": False,
        "notes": f"Digitized from manually-split panel crop {crop_path} (Figure S5 is a 2-panel figure; "
                 f"ingest_paper.py's region detector doesn't split multi-panel figures). Overlay at "
                 f"{overlay_path}. Diagnostics: {digitized['diagnostics']}. Companion survey spectrum is "
                 f"Figure S4; this is the high-resolution Cu 2p region used for the Cu(I)/Cu(0) oxidation-"
                 f"state assignment via the modified Auger parameter (paper reports 1849.1 +/- 0.1 eV, "
                 f"characteristic of Cu2O rather than metallic Cu).",
    }
    record["confidence_score"], record["confidence_breakdown"] = compute_confidence_score(record)
    problems = validate_record(record)
    if problems:
        raise ValueError(f"Figure S5{panel} failed validation: {problems}")
    out = SPECTRA_DIR / f"{safe_label}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2)
    print(f"Wrote {record['record_id']}: {len(record['x_values'])} points, "
          f"err={record['digitization_error_estimate']:.1f}, conf={record['confidence_score']:.2f}, "
          f"n_corrected={digitized['diagnostics']['n_near_vertical_columns_corrected']}")
    return record


# Panel a: Cu 2p high-res, x reversed (970->930 eV), y 0-200000 cps
xt_a = [170.0, 251.0, 332.0, 413.0, 494.0, 574.0, 655.0, 736.0]
yt_a = [68.0, 236.0, 404.0, 571.5]  # manually confirmed -- see docstring
d, ov, cp = digitize_panel("a", xt_a, yt_a, x_first=970, x_last=935, y_first=200000, y_last=50000,
                            x_tick_step=5, y_tick_step=50000)
build_record("a", d, cp, ov, Modality.XPS.value, "Binding energy (eV)", "Intensity (cps)",
              "High-resolution scan of the Cu 2p region to resolve the Cu 2p 1/2 (~952.6 eV) and Cu 2p 3/2 "
              "(~932.5 eV) spin-orbit doublet; the Cu 2p 3/2 binding energy alone doesn't distinguish "
              "Cu(I) oxide from metallic Cu(0), which is why the companion Auger spectrum (panel b) was "
              "also collected", None)

# Panel b: Cu Auger, x 905-930 eV kinetic energy, y ~10000-40000 cps
xt_b = [167.0, 228.0, 288.5, 349.5, 410.0, 471.0, 530.5, 591.0, 652.0, 712.0]
yt_b = [69.0, 162.0, 254.0, 347.0, 440.0, 532.0, 625.5]
d, ov, cp = digitize_panel("b", xt_b, yt_b, x_first=905, x_last=927.5, y_first=40000, y_last=10000,
                            x_tick_step=2.5, y_tick_step=5000)
build_record("b", d, cp, ov, Modality.XPS.value, "Kinetic energy (eV)", "Intensity (cps)",
              "Cu L3M4,5M4,5 Auger spectrum, used with the Cu 2p 3/2 binding energy (panel a) to compute "
              "the modified Auger parameter -- the paper reports 1849.1 +/- 0.1 eV, which the cited "
              "reference (Biesinger et al. 2017, 10.1002/sia.6239) identifies as characteristic of Cu2O "
              "rather than metallic Cu(0), resolving the ambiguity that Cu 2p binding energy alone leaves",
              "Figure S5a -- Cu 2p binding energy alone; Auger parameter combines both to distinguish Cu2O from metallic Cu")

print("Done.")
