"""One-off script: RSC Advances paper 10.1039/d3ra08965g (Ga-doped
Cs2AgBiBr6 double perovskite). Only 2 of this paper's 5 detected regions
are genuine continuous spectra -- Fig. 2a (XRD) and Fig. 3a (Tauc bandgap
plot) -- everything else (Fig. 2b/c/d, 3b, 4a/b) is a 2-point line/bar
chart comparing exactly 2 samples on a categorical x-axis, not a
continuous instrumental scan, so nothing to digitize there. Fig. 5 (J-V
curve) and Table 1 (solar cell params) are device-performance data, out
of scope per this project's material-property-and-spectra-only rule.

Both Fig. 2 and Fig. 3 are 2x2 / 1x2 grids respectively; panel (a) is
split out via a whitest-column/row-near-midpoint crop, same technique
used throughout this project for ACS Omega's multi-panel figures. Both
target panels have TWO curves (black = pure Cs2AgBiBr6, red = Ga-doped)
sharing one axis, separated here by --curve-color same as this project's
Figure S11/S12/S13.
"""
import json, sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, "/sessions/elegant-happy-euler/mnt/CatalysisAI/SpectraScribe-/src")
from extract_spectrum import (_find_border_box, _find_ticks, _isolate_curve_component, _linear_calibration,
                               _color_distance_mask, write_overlay, BORDER_DETECT_THRESHOLD,
                               BORDER_SEARCH_MARGIN, trace_curve_columns, paper_id_from_doi)
from schema import Modality, SourceType, ExtractionMethod, ReviewStatus, compute_confidence_score, validate_record
import numpy as np
from PIL import Image

PAPER_DIR = Path("/sessions/elegant-happy-euler/mnt/CatalysisAI/SpectraScribe-/data/papers/d3ra08965g")
CROP_DIR = PAPER_DIR / "crops"
SPECTRA_DIR = PAPER_DIR / "spectra"
SPECTRA_DIR.mkdir(exist_ok=True)
DOI = "10.1039/d3ra08965g"

manifest = json.load(open(PAPER_DIR / "manifest.json"))
citation_meta = manifest["citation_metadata"]
assert citation_meta["metadata_reviewed"]

BLACK = (0, 0, 0)
RED = (217, 10, 17)


def digitize_curve(panel_path, curve_color, x_first, x_last, x_tick_step,
                    y_first, y_last, y_tick_step, y_uncalibrated, exclude_boxes, color_tolerance=45.0):
    gray = np.array(Image.open(panel_path).convert("L"))
    border_mask = gray < BORDER_DETECT_THRESHOLD
    top, bottom, left, right = _find_border_box(border_mask)
    x_ticks, y_ticks = _find_ticks(border_mask, top, bottom, left, right)
    if len(x_ticks) < 2:
        raise ValueError(f"only {len(x_ticks)} x-ticks found in {panel_path}")

    x_slope, x_intercept, x_resid = _linear_calibration(x_ticks, x_first, x_last, x_tick_step)
    if y_uncalibrated:
        y_slope, y_intercept, y_resid = _linear_calibration([top, bottom], y_first, y_last)
    else:
        if len(y_ticks) < 2:
            raise ValueError(f"only {len(y_ticks)} y-ticks found in {panel_path}")
        y_slope, y_intercept, y_resid = _linear_calibration(y_ticks, y_first, y_last, y_tick_step)

    rgb = np.array(Image.open(panel_path).convert("RGB"))
    ink = _color_distance_mask(rgb, curve_color, color_tolerance)
    diff = rgb.astype(np.float64) - np.array(curve_color, dtype=np.float64)
    dist = np.sqrt((diff ** 2).sum(axis=-1))
    weight = np.clip(color_tolerance - dist, 0.0, None)

    for (bx1, by1, bx2, by2) in exclude_boxes:
        ink[by1:by2, bx1:bx2] = False
        weight[by1:by2, bx1:bx2] = 0.0

    interior = ink[top + BORDER_SEARCH_MARGIN: bottom - BORDER_SEARCH_MARGIN,
                    left + BORDER_SEARCH_MARGIN: right - BORDER_SEARCH_MARGIN]
    weight_interior = weight[top + BORDER_SEARCH_MARGIN: bottom - BORDER_SEARCH_MARGIN,
                              left + BORDER_SEARCH_MARGIN: right - BORDER_SEARCH_MARGIN]
    curve_mask = _isolate_curve_component(interior)

    trace = trace_curve_columns(curve_mask, weight_interior, top, left, x_slope, x_intercept,
                                 y_slope, y_intercept, y_resid, x_first, x_last,
                                 curve_color=curve_color, color_tolerance=color_tolerance)
    diag = {"n_x_ticks_used": len(x_ticks), "n_y_ticks_used": len(y_ticks) if not y_uncalibrated else "uncalibrated",
            "x_calibration_residual_std": x_resid, "y_calibration_residual_std": y_resid,
            **trace["diagnostics"], "border_box_px": {"top": top, "bottom": bottom, "left": left, "right": right}}
    d = {"x_values": trace["x_values"], "y_values": trace["y_values"],
         "digitization_error_estimate": trace["digitization_error_estimate"], "diagnostics": diag,
         "_curve_mask": curve_mask, "_border": (top, bottom, left, right)}
    return d


def build_record(safe_label, source_location, curve_label, digitized, panel_path, modality,
                  material_formula, material_description, measurement_purpose, x_axis, y_axis,
                  baseline_material, extra_note=""):
    overlay_path = str(SPECTRA_DIR / f"{safe_label}_overlay.png")
    write_overlay(panel_path, digitized, overlay_path)
    record = {
        "record_id": f"d3ra08965g_{safe_label.lower()}",
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
        "curator": "SpectraScribe (extract_rsc_d3ra08965g.py, one-off panel split + color-separated trace)",
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_name": citation_meta.get("journal") or "unknown",
        "source_url": f"https://doi.org/{DOI}",
        "doi": manifest["doi"],
        "source_location": f"{source_location} ({curve_label})",
        "license": manifest["license"],
        "citation": citation_meta["citation"],
        "peer_reviewed": True,
        "open_access": True,
        "cross_validated": False,
        "notes": f"Digitized from manually-split panel crop {panel_path} (this figure is a multi-panel "
                 f"grid; ingest_paper.py's region detector doesn't split panels). Overlay at "
                 f"{overlay_path}. Diagnostics: {digitized['diagnostics']}." + extra_note,
    }
    record["confidence_score"], record["confidence_breakdown"] = compute_confidence_score(record)
    problems = validate_record(record)
    if problems:
        raise ValueError(f"{safe_label} failed validation: {problems}")
    out = SPECTRA_DIR / f"{safe_label}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2)
    print(f"Wrote {record['record_id']}: {len(record['x_values'])} points, "
          f"err={record['digitization_error_estimate']:.3g}, conf={record['confidence_score']:.2f}")
    return record


if __name__ == "__main__":
    # --- Figure 2a: XRD pattern, black=pure, red=Ga-doped, 2theta 10-45 deg ---
    fig2a_path = str(CROP_DIR / "_panel_Fig2a.png")
    fig2a_exclude = [(190, 40, 490, 235),   # inset (zoomed (400) peak, 29.4-33.6 deg)
                      (685, 10, 1005, 115)]  # legend box
    for label, color, tol in [("black_pure", BLACK, 130.0), ("red_ga_doped", RED, 140.0)]:
        d = digitize_curve(fig2a_path, color, x_first=10, x_last=45, x_tick_step=2.5,
                            y_first=1.0, y_last=0.0, y_tick_step=None, y_uncalibrated=True,
                            exclude_boxes=fig2a_exclude, color_tolerance=tol)
        material = "Cs2AgBiBr6" if "pure" in label else "Cs2Ag0.95Ga0.05BiBr6"
        build_record(f"Figure_2a_{label}", "Figure 2a", material, d, fig2a_path, Modality.XRD.value,
                     material,
                     f"{material} double perovskite solar cell absorber layer, synthesized via sol-gel method",
                     "Powder XRD pattern to confirm the cubic double-perovskite phase and quantify the "
                     "effect of Ga3+ substitution on lattice parameters (peak positions match the Fm-3m "
                     "space group; inset shows the (400) peak shift on Ga substitution)",
                     "2-theta (degree)", "Intensity (a.u., relative pixel scale -- source prints no y-axis "
                     "tick numbers)",
                     None if "pure" in label else "Cs2AgBiBr6 (pure, undoped)",
                     extra_note=" No y-axis tick numbers are printed on this figure (checked the full page, "
                     "not just the crop) -- y_values are on a relative 0-1 pixel scale, not calibrated counts.")

    # --- Figure 3a: Tauc plot ((ahv)^2 vs hv), black=pure, red=Ga-doped ---
    # Each curve has its own straight tangent/extrapolation line (same color, not
    # dashed) used by the authors to read off the bandgap x-intercept -- unlike this
    # project's Figure S10 (dashed tangents), here tangent and real curve share both
    # color AND line style, so they cannot be told apart by masking alone. Restricting
    # the traced x-range to where the real curve is unambiguous (verified per-curve
    # against the overlay, see below) rather than guessing.
    fig3a_path = str(CROP_DIR / "_panel_Fig3a.png")
    fig3a_exclude = [(265, 105, 655, 215)]  # legend box
    for label, color, x_last, tol in [("black_pure", BLACK, 2.5, 130.0), ("red_ga_doped", RED, 2.5, 140.0)]:
        d = digitize_curve(fig3a_path, color, x_first=0.0, x_last=x_last, x_tick_step=0.25,
                            y_first=6e5, y_last=0.0, y_tick_step=0.5e5, y_uncalibrated=False,
                            exclude_boxes=fig3a_exclude, color_tolerance=tol)

        truncate_note = ""
        if label == "black_pure":
            # Standalone plot (plotted independently of the source image, exactly the
            # check this project always runs before trusting a digitized curve) showed
            # a sharp spurious dip at hv~1.965-2.09 eV: y dropped from 177418 to 86248
            # then jumped to 248218 one point later -- physically impossible for a
            # smooth absorption edge. Root cause: this curve's straight black tangent
            # line (same color, solid, not dashed -- see module docstring) crosses the
            # real curve in exactly this range, and color-only masking can't tell which
            # of the two same-colored lines is which once they're both present in a
            # column. Rather than guess which line the algorithm followed past the
            # crossing (a wrong-but-smooth continuation would be a WORSE, silent error
            # than this visible jump), the data is truncated at the last point before
            # the jump (hv=1.9647) -- an honest reported gap, not fabricated across,
            # same policy as this project's occlusion-gap handling elsewhere.
            cutoff = next(i for i, x in enumerate(d["x_values"]) if x > 1.9647)
            d["x_values"] = d["x_values"][:cutoff]
            d["y_values"] = d["y_values"][:cutoff]
            d["diagnostics"]["truncated_at_hv"] = 1.9647
            d["diagnostics"]["truncation_reason"] = "tangent-line/curve color ambiguity past this point (see notes)"
            truncate_note = (" TRUNCATED at hv=1.9647 eV (originally traced to ~2.3 eV): the tangent-line "
                              "crossing beyond this point produced a spurious jump on the standalone "
                              "verification plot (177418 -> 86248 -> 248218 across 2 points) -- see "
                              "diagnostics.truncation_reason. The excluded region is not reported as data.")

        material = "Cs2AgBiBr6" if "pure" in label else "Cs2Ag0.95Ga0.05BiBr6"
        build_record(f"Figure_3a_{label}", "Figure 3a", material, d, fig3a_path, Modality.UV_VIS_TAUC.value,
                     material,
                     f"{material} double perovskite thin film",
                     "Tauc plot analysis (direct-allowed transition, n=2) of UV-vis absorption to extract "
                     "the optical bandgap; paper reports a bandgap reduction on Ga substitution",
                     "hv (eV)", "(alpha*hv)^2 (eV/cm)^2",
                     None if "pure" in label else "Cs2AgBiBr6 (pure, undoped)",
                     extra_note=" This curve also has a straight linear-extrapolation tangent line in the "
                     "source figure (same color, solid like the real curve, not dashed) used by the authors "
                     "to read off the bandgap x-intercept -- NOT real data. Checked the overlay for "
                     "tangent-line contamination before marking reviewed." + truncate_note)

    print("Done.")
