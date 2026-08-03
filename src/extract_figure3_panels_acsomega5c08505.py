"""One-off script: Figure 3 is a 4-panel figure (a,b,c,d) that ingest_paper.py's
region detector captured as a single crop (multi-panel splitting is a documented
Phase 1 limitation, not built). Panels were manually split via a whitespace-gap
detector (see chat) into data/papers/acsomega_5c08505/crops/_panels/Figure_3{a,b,c,d}.png.
This script digitizes each panel directly via extract_spectrum's digitize()/
write_overlay(), bypassing the manifest-region-lookup CLI path since these
sub-panel crops aren't manifest regions."""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, "/sessions/elegant-happy-euler/mnt/CatalysisAI/SpectraScribe-/src")
from extract_spectrum import digitize, write_overlay, _linear_calibration, _find_border_box, _find_ticks, BORDER_DETECT_THRESHOLD
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

def digitize_panel(panel, x_first, x_last, y_first, y_last, y_uncal, n_xticks_use=None, x_tick_step=None):
    path = str(PANEL_DIR / f"Figure_3{panel}.png")
    gray = np.array(Image.open(path).convert("L"))
    border_mask = gray < BORDER_DETECT_THRESHOLD
    top, bottom, left, right = _find_border_box(border_mask)
    x_ticks, y_ticks = _find_ticks(border_mask, top, bottom, left, right)
    if n_xticks_use:
        x_ticks = x_ticks[:n_xticks_use]
    # Reuse digitize()'s internal logic by calling the pieces directly since we need
    # custom tick-list truncation (panel b) that the CLI path doesn't expose.
    from extract_spectrum import _isolate_curve_component, BORDER_SEARCH_MARGIN, CURVE_DARK_THRESHOLD, trace_curve_columns
    x_slope, x_intercept, x_resid = _linear_calibration(x_ticks, x_first, x_last, x_tick_step)
    if y_uncal:
        y_slope, y_intercept, y_resid = _linear_calibration([top, bottom], y_first, y_last)
    else:
        y_slope, y_intercept, y_resid = _linear_calibration(y_ticks, y_first, y_last)
    ink = gray < CURVE_DARK_THRESHOLD
    weight = np.clip(CURVE_DARK_THRESHOLD - gray.astype(np.float64), 0.0, None)
    interior = ink[top+BORDER_SEARCH_MARGIN:bottom-BORDER_SEARCH_MARGIN, left+BORDER_SEARCH_MARGIN:right-BORDER_SEARCH_MARGIN]
    weight_interior = weight[top+BORDER_SEARCH_MARGIN:bottom-BORDER_SEARCH_MARGIN, left+BORDER_SEARCH_MARGIN:right-BORDER_SEARCH_MARGIN]
    curve_mask = _isolate_curve_component(interior)
    # Shared tracer (near-vertical-peak fix, sub-pixel weighted centroid,
    # median de-noising) -- see trace_curve_columns()'s docstring in
    # extract_spectrum.py. Used to be a duplicated loop here that only got
    # the near-vertical fix, not the later weighting/smoothing fixes, until
    # those had to be re-added a second time; now there's one copy to fix.
    trace = trace_curve_columns(curve_mask, weight_interior, top, left, x_slope, x_intercept,
                                 y_slope, y_intercept, y_resid, x_first, x_last)
    x_values, y_values = trace["x_values"], trace["y_values"]
    diag = {"n_x_ticks_found": len(x_ticks), "n_y_ticks_found": len(y_ticks),
            "x_calibration_residual_std": x_resid, "y_calibration_residual_std": y_resid,
            **trace["diagnostics"], "border_box_px": {"top": top, "bottom": bottom, "left": left, "right": right}}
    d = {"x_values": x_values, "y_values": y_values,
         "digitization_error_estimate": trace["digitization_error_estimate"],
         "diagnostics": diag, "_curve_mask": curve_mask, "_border": (top, bottom, left, right)}
    overlay_path = str(SPECTRA_DIR / f"Figure_3{panel}_overlay.png")
    write_overlay(path, d, overlay_path)
    return d, overlay_path, path

def build_record(panel, digitized, crop_path, overlay_path, modality, x_axis, y_axis,
                  material_formula, material_description, measurement_purpose, baseline_material, notes_extra=""):
    safe_label = f"Figure_3{panel}"
    record = {
        "record_id": f"acsomega_5c08505_{safe_label.lower()}",
        "material_formula": material_formula,
        "material_description": material_description,
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
        "curator": "SpectraScribe (extract_figure3_panels.py, one-off multi-panel split + algorithmic trace)",
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_name": citation_meta.get("journal") or "unknown",
        "source_url": f"https://doi.org/{DOI}",
        "doi": manifest["doi"],
        "source_location": f"Figure 3{panel}",
        "license": manifest["license"],
        "citation": citation_meta["citation"],
        "peer_reviewed": True,
        "open_access": True,
        "cross_validated": False,
        "notes": f"Digitized from manually-split panel crop {crop_path} (Figure 3 is a 4-panel "
                 f"figure; ingest_paper.py's region detector doesn't split multi-panel figures -- "
                 f"see docs). Overlay at {overlay_path}. Diagnostics: {digitized['diagnostics']}. {notes_extra}",
    }
    record["confidence_score"], record["confidence_breakdown"] = compute_confidence_score(record)
    problems = validate_record(record)
    if problems:
        raise ValueError(f"Figure 3{panel} failed validation: {problems}")
    out = SPECTRA_DIR / f"{safe_label}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2)
    print(f"Wrote {record['record_id']}: {len(record['x_values'])} points, "
          f"err={record['digitization_error_estimate']:.2f}, conf={record['confidence_score']:.2f}")
    return record

MATERIAL_DESC_POWDER = "TiO2/Cu2O powder sample, prepared via homogeneous synthesis with 60 min reaction time (not the coated-on-substrate samples used elsewhere in this paper)"

# Panel a: XRD wide range, 5-85 deg 2theta
d, ov, cp = digitize_panel("a", x_first=20, x_last=80, y_first=1.0, y_last=0.0, y_uncal=True)
build_record("a", d, cp, ov, Modality.XRD.value, "2theta (deg)",
             "Intensity (a.u., relative pixel scale 0-1 -- source figure prints no y-axis tick values)",
             "TiO2/Cu2O", MATERIAL_DESC_POWDER,
             "Identify crystal phases via XRD peak positions/indices (110),(111),(200),(220),(311),(222) -- confirms cubic Cu2O crystal structure",
             None)

# Panel b: XRD zoomed 25-50 deg -- use only first 5 evenly-spaced detected ticks (6th is anomalous, see chat)
d, ov, cp = digitize_panel("b", x_first=25, x_last=45, y_first=1.0, y_last=0.0, y_uncal=True, n_xticks_use=5)
build_record("b", d, cp, ov, Modality.XRD.value, "2theta (deg)",
             "Intensity (a.u., relative pixel scale 0-1 -- source figure prints no y-axis tick values)",
             "TiO2/Cu2O", MATERIAL_DESC_POWDER,
             "Zoomed view (25-50 deg 2theta) of the same XRD pattern as Figure 3a, to show the (111)/(200) peak region in more detail",
             None,
             notes_extra="x-axis calibrated from only the first 5 of 6 auto-detected ticks (25,30,35,40,45); "
                          "the 6th detected tick did not fit the even-spacing pattern of the others (likely a "
                          "minor gridline near the border, not the true '50' major tick) and was excluded rather "
                          "than risk a bad calibration anchor.")

# Panel c: Raman, full range, fully calibrated both axes
d, ov, cp = digitize_panel("c", x_first=200, x_last=1400, y_first=8000, y_last=0, y_uncal=False)
build_record("c", d, cp, ov, Modality.RAMAN.value, "Raman shift (cm-1)", "Intensity (a.u.)",
             "TiO2/Cu2O", MATERIAL_DESC_POWDER,
             "Identify Cu2O Raman-active bands on the powder sample; red asterisks in the source figure mark bands the authors attribute to a secondary/impurity phase rather than Cu2O itself",
             "Figure S2 -- Raman spectrum of bare glass substrate, same DOI (different physical sample -- powder vs. glass-substrate coating -- but same baseline-subtraction logic for identifying which peaks are substrate/impurity artifacts)")

# Panel d: Raman, same x range, y rescaled to 0-700 to show weak features
d, ov, cp = digitize_panel("d", x_first=200, x_last=1400, y_first=700, y_last=0, y_uncal=False)
build_record("d", d, cp, ov, Modality.RAMAN.value, "Raman shift (cm-1)", "Intensity (a.u.)",
             "TiO2/Cu2O", MATERIAL_DESC_POWDER,
             "Same measurement as Figure 3c, re-plotted at a smaller y-axis range (0-700 vs 0-8000) specifically to make the weaker secondary peaks (146, 215, 412, 634 cm-1) readable -- the intense 142 cm-1 peak is clipped off-scale in this view",
             "Figure 3c -- same sample, same measurement, wider y-axis range")

print("Done.")
