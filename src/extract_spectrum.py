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

Multi-curve figures (e.g. Figure S11: two curves, near-black and orange;
Figure S12: four curves, all shades of orange) are handled by running
this script once PER CURVE -- --curve-color isolates only the target
curve's color and the connected-component filter ignores everything
else on the plot, one curve at a time, never attempting to trace every
curve in a single pass. --curve-label is required in this case so each
curve gets its own record_id/filename/source_location and results don't
collide or silently overwrite each other.

Known limitations, stated up front:
  - Curves that visually cross or run very close together can blend at
    the crossing point into an intermediate color neither curve's
    tolerance matches, producing a small gap in the trace right at the
    crossing -- the connected-component width filter can still keep the
    rest of the curve if the gap is short relative to the plot width,
    but a long shared/overlapping run isn't handled. Check the overlay.
  - Assumes ticks are evenly spaced between the first and last detected
    tick (true for every figure in this project's real test paper, and
    for the overwhelming majority of scientific plots, but not
    universally true of every possible axis -- e.g. a log axis would
    break this). No log-axis support yet.
  - Assumes a roughly white/light plot background and a dark
    (black/near-black) curve -- true for this project's real test
    figures, not guaranteed for a colored-curve or dark-theme plot.

Usage (single-curve figure):
    python extract_spectrum.py --doi 10.xxxx/yyyy --label "Figure S2" \\
        --x-first-tick 100 --x-last-tick 1500 \\
        --y-first-tick 1000 --y-last-tick 0 \\
        --modality Raman --x-axis "Raman shift (cm-1)" --y-axis "Intensity (a.u.)" \\
        --material-formula "SiO2" --material-description "bare glass substrate (blank)" \\
        --measurement-purpose "confirm glass substrate's own Raman signal doesn't overlap sample peaks" \\
        --baseline-material null
    python extract_spectrum.py --doi 10.xxxx/yyyy --label "Figure S2" --mark-reviewed

Usage (multi-curve figure -- one call per curve, e.g. Figure S11):
    python extract_spectrum.py --doi 10.xxxx/yyyy --label "Figure S11" \\
        --curve-label "Before adsorption step" --curve-color "0,0,0" \\
        --x-first-tick 350 --x-last-tick 600 --y-first-tick 0.30 --y-last-tick 0.00 \\
        --modality UV-Vis --x-axis "Wavelength (nm)" --y-axis "Absorbance (-)" \\
        --material-formula "methyl orange" --material-description "methyl orange dye solution, before adsorption onto TiO2/Cu2O coating" \\
        --measurement-purpose "baseline dye concentration prior to adsorption step" --baseline-material null
    python extract_spectrum.py --doi 10.xxxx/yyyy --label "Figure S11" \\
        --curve-label "After adsorption step" --curve-color "255,132,0" \\
        --x-first-tick 350 --x-last-tick 600 --y-first-tick 0.30 --y-last-tick 0.00 \\
        --modality UV-Vis --x-axis "Wavelength (nm)" --y-axis "Absorbance (-)" \\
        --material-formula "methyl orange" --material-description "methyl orange dye solution, after adsorption onto TiO2/Cu2O coating" \\
        --measurement-purpose "dye concentration after adsorption, to compute adsorbed fraction" \\
        --baseline-material "Figure S11 (Before adsorption step) -- same figure, same DOI"
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


DARK_THRESHOLD = 100        # grayscale value below which a pixel counts as generic "ink" -- NOT used
                             # for curve tracing (see CURVE_DARK_THRESHOLD) or border/tick detection
                             # (see BORDER_DETECT_THRESHOLD), both of which need looser thresholds;
                             # kept as the strict baseline for anything else.
BORDER_DETECT_THRESHOLD = 200  # grayscale value used ONLY for finding the border box and tick marks.
                             # Looser than DARK_THRESHOLD on purpose -- found necessary by hitting it:
                             # this project's real Figure S3 draws its TOP border line in a lighter
                             # gray (~gray 93-based row only registers below ~200, not below 100) while
                             # its bottom/left/right borders are solid black -- an inconsistency within
                             # the SAME figure, not just across figures. At DARK_THRESHOLD=100, border
                             # detection missed the true top border entirely and locked onto some other
                             # dense-but-wrong row instead, which silently clipped the tallest peak in
                             # the whole spectrum (146 cm-1) out of the interior region entirely -- no
                             # error, just missing data, caught only by checking traced x-range against
                             # the actual peak labels, same as every other real bug in this file. A
                             # border/tick line is identifiable by spanning nearly the FULL width or
                             # height of the image regardless of its exact shade, so a looser threshold
                             # here is safe: real content (curves, text) essentially never also spans
                             # the full image width/height, so there's no realistic risk of the looser
                             # threshold picking up something else as "the border."
CURVE_DARK_THRESHOLD = 180  # grayscale value below which a pixel counts as "curve ink" in the default
                             # (no --curve-color) single-black-curve mode. Looser than DARK_THRESHOLD
                             # on purpose -- found necessary by hitting it: a real near-black curve
                             # (Figure S11's "Before adsorption step" line) has long, nearly-flat
                             # stretches where anti-aliasing spreads the line's darkness thin enough
                             # that its darkest pixel in a column only reaches gray~130-150, well
                             # above DARK_THRESHOLD=100. At threshold=100 the curve fragmented into
                             # ~70 disconnected pieces and the width filter below kept only the
                             # steepest (peak) region, silently dropping both flat wings (350-394nm
                             # and 536-600nm) while still looking like a plausible, complete curve --
                             # confirmed by checking traced x-range against the actual figure, not by
                             # inspecting the overlay alone. At 170-180 the same curve traces as one
                             # single component spanning the full plot width. Re-verified this doesn't
                             # break the original single-curve pilot (Figure S2): its curve is already
                             # a full-width single component at threshold=100, and stays full-width
                             # (just a couple pixels thicker) at 180 too -- so loosening this was a
                             # strict improvement on real test data, not a tradeoff made blind.
BORDER_SEARCH_MARGIN = 3    # px excluded on each side of a detected border line when scanning for ticks/curve
TICK_BAND_PX = 7            # how far outside the border to look for tick marks
TICK_MIN_DARK_FRACTION = 0.5  # fraction of the tick band that must be dark to count as a tick
MIN_CURVE_SEGMENT_PX = 45  # a connected component must span at least this many pixels to count as
                            # real curve data rather than a text label -- calibrated against this
                            # project's real peak-position labels ("557", "793", "1101" on Figure
                            # S2), each ~20-21px wide; set with roughly 2x margin above that. An
                            # absolute pixel count, not a fraction of plot width, on purpose -- see
                            # _isolate_curve_component()'s docstring for why a width-fraction filter
                            # was wrong (discarded real, separate curve segments on either side of a
                            # genuine occlusion gap).


def _find_border_box(dark: np.ndarray) -> tuple:
    """Locates the plot's rectangular axis border: the row/column with
    the most dark pixels, searched separately in each side's own outer
    QUARTER of the image (top in rows [0, h/4), bottom in [3h/4, h), same
    idea for left/right columns). Returns (top, bottom, left, right)
    pixel coordinates. Assumes exactly one clean rectangular border,
    true of every figure in this project's real test data.

    Restricted to a quarter, not a half, on purpose -- found necessary
    by hitting it on a real figure (S3) whose x-axis has an in-plot
    annotation ('634', a peak label with a leader line) sitting past the
    horizontal midpoint of the image: at BORDER_DETECT_THRESHOLD's looser
    threshold (needed to catch this same figure's unusually light top
    border line -- see that constant's docstring), the annotation's
    column tied the real right border's dark-pixel count, and argmax's
    first-occurrence tiebreak picked the annotation instead, silently
    cutting off roughly a third of the real x-axis range. A quarter-width
    restriction keeps genuine border lines (always right at the plot's
    edge) in scope while excluding in-plot content like this, which sits
    well inside the middle of the image regardless of threshold."""
    h, w = dark.shape
    row_counts = dark.sum(axis=1)
    col_counts = dark.sum(axis=0)
    top = int(np.argmax(row_counts[: h // 4]))
    bottom = int(3 * h // 4 + np.argmax(row_counts[3 * h // 4 :]))
    left = int(np.argmax(col_counts[: w // 4]))
    right = int(3 * w // 4 + np.argmax(col_counts[3 * w // 4 :]))
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
    (scipy.ndimage.label) and keeping every component whose bounding box
    spans at least MIN_CURVE_SEGMENT_PX -- measured directly against this
    project's real peak-label text (Figure S2's "557"/"793"/"1101": each
    digit group spans ~20-21px regardless of plot size), so the threshold
    is set well above that (with margin) rather than as a fraction of
    plot width.

    This matters, not just as a style choice: an earlier version required
    a component to span >=50% of the FULL plot width to be kept, which
    silently discarded genuine curve data whenever the true curve was
    broken into several separate pieces by real gaps -- found on this
    project's real Figure S11, where the 'Before adsorption' curve is
    partly occluded by the 'After adsorption' curve drawn on top of it
    where the two nearly coincide (see extract_spectrum usage notes/
    schema baseline_material field): the visible left wing, peak, and
    right wing each individually span under 50% of the plot, so a
    width-FRACTION filter kept none of them, while the label-sized
    absolute-pixel filter correctly keeps all three as real curve data.
    A genuine occlusion gap between kept segments is still a real,
    honest gap in the output (not fabricated across) -- it just isn't
    also punished by discarding the segments on either side of it."""
    labeled, n = ndimage.label(dark_interior, structure=np.ones((3, 3)))
    keep = np.zeros_like(dark_interior, dtype=bool)
    for label_id in range(1, n + 1):
        cols = np.where((labeled == label_id).any(axis=0))[0]
        if len(cols) == 0:
            continue
        span = cols.max() - cols.min()
        if span >= MIN_CURVE_SEGMENT_PX:
            keep |= labeled == label_id
    return keep


def _linear_calibration(pixel_positions: list, first_value: float, last_value: float,
                         tick_step: float = None) -> tuple:
    """Fits pixel -> data-value as a straight line through the detected
    ticks. Returns (slope, intercept, residual_std).

    Two modes:

    tick_step=None (default): assumes the N detected tick pixels are N
    CONSECUTIVE evenly-spaced ticks running from first_value to
    last_value, via np.linspace -- correct as long as tick detection
    found every tick with none missing in the middle.

    tick_step=<value>: does NOT assume the detected pixels are
    consecutive. Instead computes each pixel's value from a straight
    line through (first pixel, first_value) and (last pixel, last_value)
    directly, then SNAPS that estimate to the nearest real multiple of
    tick_step -- self-consistent, so a tick that got missed in the
    middle of the detected list (found necessary on this project's real
    Figure S3: the curve itself crosses over 2 of the 13 real tick
    positions, hiding them from _find_ticks entirely, so the 11 ticks
    that WERE found are not 11 consecutive values) doesn't silently
    shift every value after the gap. Caught by comparing traced peak
    positions against the paper's own printed peak labels (146, 215,
    412, 634 cm-1): the naive linspace assumption was off by up to 19
    units on some peaks; snap-to-tick_step brought that back down to a
    few units, consistent with every other figure in this project."""
    pixel_positions = np.array(pixel_positions, dtype=np.float64)
    if tick_step is None:
        n = len(pixel_positions)
        assumed_values = np.linspace(first_value, last_value, n)
    else:
        first_px, last_px = pixel_positions[0], pixel_positions[-1]
        rough_slope = (last_value - first_value) / (last_px - first_px)
        rough_estimate = first_value + (pixel_positions - first_px) * rough_slope
        assumed_values = np.round((rough_estimate - first_value) / tick_step) * tick_step + first_value
    slope, intercept = np.polyfit(pixel_positions, assumed_values, 1)
    predicted = slope * pixel_positions + intercept
    residual_std = float(np.std(assumed_values - predicted))
    return float(slope), float(intercept), residual_std


def _color_distance_mask(rgb: np.ndarray, target_color: tuple, tolerance: float) -> np.ndarray:
    """Boolean mask of pixels within `tolerance` Euclidean RGB distance of
    target_color. Used instead of the plain darkness threshold when a
    figure has more than one curve distinguished by color (this project's
    real Figure S11: a near-black 'Before adsorption' curve and an
    orange(255,132,0) 'After adsorption' curve, both with a legend; S12:
    four curves that are all shades of orange, distinguished by
    saturation/lightness rather than hue). Sampling exact target colors
    (not eyeballed) matters here: legend swatches and curve ink for the
    same series are the same RGB in a vector-drawn/exported plot, so a
    tight-ish tolerance cleanly separates same-figure curves that are
    visually close (see S12's four orange shades) without needing hue-only
    matching, which would confuse them."""
    diff = rgb.astype(np.float64) - np.array(target_color, dtype=np.float64)
    dist = np.sqrt((diff ** 2).sum(axis=-1))
    return dist <= tolerance


def digitize(crop_path: str, x_first_tick: float, x_last_tick: float,
             y_first_tick: float, y_last_tick: float,
             curve_color: tuple = None, color_tolerance: float = 45.0,
             y_uncalibrated: bool = False, x_tick_step: float = None,
             y_tick_step: float = None, exclude_boxes: list = None) -> dict:
    """Runs the full pipeline against one crop image. Returns a dict with
    x_values, y_values, digitization_error_estimate, and diagnostic info
    (tick counts found, calibration residuals) for the caller to inspect
    before deciding whether to trust the result at all.

    curve_color: when given (an (R,G,B) tuple), isolates ONLY the curve
    matching that color -- see _color_distance_mask -- for multi-curve
    figures. When None (default), falls back to plain darkness
    thresholding, for the common single-black-curve case (e.g. Figure
    S2). Axis border/tick detection ALWAYS uses darkness, regardless --
    the axis frame is black no matter what color the data curve is.

    y_uncalibrated: some real published spectra (this project's own
    Figure S3 among them) print NO y-axis tick marks or numbers at all --
    just an axis label like 'Intensity (a.u.)', because the absolute
    scale genuinely isn't meaningful (baseline-offset, arbitrary units).
    Confirmed by checking the full page render, not assumed from the
    crop alone -- ruled out that region-detection had cropped real tick
    labels out. When true, y_first_tick/y_last_tick are treated as the
    values at the border's TOP and BOTTOM pixel rows directly (not tick
    positions), producing a relative 0-1-style scale rather than a false
    claim of calibrated intensity units -- the caller is responsible for
    labeling y_axis accordingly (e.g. '...relative pixel scale, source
    prints no y-axis ticks') so nobody downstream mistakes this for a
    real calibrated intensity axis.

    exclude_boxes: list of (x1, y1, x2, y2) pixel rectangles, in the full
    crop image's own coordinates, to blank out of the curve-ink mask
    before isolation. Needed for legend boxes: a legend's color swatch
    line is drawn in the exact same RGB as its curve (by construction --
    that's the point of a legend), so color-distance isolation picks it
    up as a second same-colored connected component. If the swatch is
    wide enough to pass MIN_CURVE_SEGMENT_PX and shares any x-columns
    with the real curve, the per-column row-average below silently
    blends the real curve's row with the legend swatch's row in those
    columns, corrupting the trace with a flat, wrong plateau at the
    legend's y-position. Caught for real on this project's own Figure 6
    (glass-substrate curve, ~90% transmittance) which got a flat wrong
    segment at 68.29% for wavelengths 737-810 nm -- exactly where the
    'glass substrate' legend entry's swatch line sits. Verify a legend
    box's pixel bounds don't overlap any real curve data before excluding
    it (here they didn't: the legend sits at y<49% transmittance in a
    wavelength range where every real curve is already above 50%)."""
    img_l = Image.open(crop_path).convert("L")
    gray = np.array(img_l)
    border_mask = gray < BORDER_DETECT_THRESHOLD

    top, bottom, left, right = _find_border_box(border_mask)
    x_ticks, y_ticks = _find_ticks(border_mask, top, bottom, left, right)
    if len(x_ticks) < 2:
        raise ValueError(
            f"Found only {len(x_ticks)} x-ticks -- need at least 2 to calibrate. This figure's tick "
            f"marks may not match the expected style (short marks just outside a rectangular border); "
            f"check the crop manually."
        )
    if not y_uncalibrated and len(y_ticks) < 2:
        raise ValueError(
            f"Found only {len(y_ticks)} y-ticks -- need at least 2 to calibrate. If this figure genuinely "
            f"has no y-axis tick marks (check the full page, not just the crop, to be sure it's not a "
            f"cropping artifact), pass y_uncalibrated=True instead."
        )

    x_slope, x_intercept, x_resid = _linear_calibration(x_ticks, x_first_tick, x_last_tick, x_tick_step)
    if y_uncalibrated:
        y_slope, y_intercept, y_resid = _linear_calibration([top, bottom], y_first_tick, y_last_tick)
    else:
        y_slope, y_intercept, y_resid = _linear_calibration(y_ticks, y_first_tick, y_last_tick, y_tick_step)

    if curve_color is not None:
        rgb = np.array(Image.open(crop_path).convert("RGB"))
        ink = _color_distance_mask(rgb, curve_color, color_tolerance)
    else:
        ink = gray < CURVE_DARK_THRESHOLD

    if exclude_boxes:
        for (bx1, by1, bx2, by2) in exclude_boxes:
            ink[by1:by2, bx1:bx2] = False

    interior = ink[top + BORDER_SEARCH_MARGIN : bottom - BORDER_SEARCH_MARGIN,
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
        hint = (f"no pixels matched curve_color={curve_color} within tolerance={color_tolerance} -- "
                f"try sampling the exact RGB again or widening --color-tolerance" if curve_color is not None
                else "check DARK_THRESHOLD against the crop image")
        raise ValueError(f"No curve pixels found after isolating the largest connected component -- {hint}.")

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

    # --- Multi-curve figures: one curve per run, never all at once ---
    # A figure with more than one curve (e.g. this project's real Figure
    # S11: 'Before adsorption step' in near-black, 'After adsorption
    # step' in orange; Figure S12: four time points, all shades of
    # orange) is handled by running this script once per curve, isolating
    # only that curve's color and ignoring everything else on the plot --
    # never attempting to trace every curve in one pass. --curve-label
    # is required whenever a figure has more than one curve specifically
    # so record_id/filenames/source_location stay distinct per curve
    # instead of colliding (see main()'s slug construction below).
    parser.add_argument("--curve-label", help='Which curve this run is for, e.g. "Before adsorption step" -- required for multi-curve figures. Appended to record_id/filenames/source_location so curves from the same figure never collide.')
    parser.add_argument("--curve-color", help='"R,G,B" of the target curve, sampled from the actual crop image (e.g. from its legend swatch) -- required when a figure has more than one curve. Omit for single-black-curve figures.')
    parser.add_argument("--color-tolerance", type=float, default=45.0, help="Euclidean RGB distance a pixel may be from --curve-color and still count as this curve.")
    parser.add_argument("--y-uncalibrated", action="store_true", help="This figure prints no y-axis tick marks/numbers at all (verified against the full page, not just the crop) -- treat --y-first-tick/--y-last-tick as the values at the plot border's top/bottom pixel rows instead of requiring real tick detection.")
    parser.add_argument("--x-tick-step", type=float, help="Data units between consecutive x-axis ticks (e.g. 100). Use when the curve itself crosses over some tick marks, hiding them from detection -- without this, detected-but-non-consecutive ticks get silently mis-assigned to consecutive values.")
    parser.add_argument("--y-tick-step", type=float, help="Same as --x-tick-step, for the y-axis.")
    parser.add_argument("--exclude-box", action="append", default=[],
                         help='"x1,y1,x2,y2" pixel rectangle (in the crop image\'s own coordinates) to blank '
                              'out before curve isolation -- use for legend boxes when the legend swatch color '
                              'matches --curve-color (it will otherwise get traced as a second curve segment '
                              'and corrupt columns it shares with the real curve). Repeatable.')
    args = parser.parse_args()

    exclude_boxes = []
    for spec in args.exclude_box:
        parts = [p.strip() for p in spec.split(",")]
        if len(parts) != 4:
            raise SystemExit(f"[ERROR] --exclude-box must be 'x1,y1,x2,y2', got {spec!r}")
        exclude_boxes.append(tuple(int(p) for p in parts))

    paper_id = paper_id_from_doi(args.doi)
    paper_dir = Path(args.data_root) / "papers" / paper_id
    manifest_path = paper_dir / "manifest.json"
    label_slug = args.label.replace(" ", "_").replace(".", "")
    curve_slug = None
    if args.curve_label:
        curve_slug = "".join(c if c.isalnum() else "_" for c in args.curve_label).strip("_")
        while "__" in curve_slug:
            curve_slug = curve_slug.replace("__", "_")
    safe_label = f"{label_slug}_{curve_slug}" if curve_slug else label_slug

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
    curve_color = None
    if args.curve_color:
        parts = [p.strip() for p in args.curve_color.split(",")]
        if len(parts) != 3:
            raise SystemExit(f"[ERROR] --curve-color must be 'R,G,B', got {args.curve_color!r}")
        curve_color = tuple(int(p) for p in parts)

    try:
        digitized = digitize(crop_path, args.x_first_tick, args.x_last_tick, args.y_first_tick, args.y_last_tick,
                              curve_color=curve_color, color_tolerance=args.color_tolerance,
                              y_uncalibrated=args.y_uncalibrated,
                              x_tick_step=args.x_tick_step, y_tick_step=args.y_tick_step,
                              exclude_boxes=exclude_boxes)
    except ValueError as e:
        raise SystemExit(f"[ERROR] {e}")

    write_overlay(crop_path, digitized, str(overlay_path))

    baseline = None if args.baseline_material.strip().lower() == "null" else args.baseline_material
    source_location = f"{region['label']} ({args.curve_label})" if args.curve_label else region["label"]

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
        "source_location": source_location,
        "license": manifest["license"],
        "citation": citation_meta["citation"],
        "peer_reviewed": True,
        "open_access": True,
        "cross_validated": False,
        "notes": f"Digitized algorithmically from {crop_path}; overlay at {overlay_path}. "
                 f"curve_color={curve_color}, color_tolerance={args.color_tolerance}"
                 + (f", exclude_boxes={exclude_boxes} (legend/annotation regions blanked out of the curve-ink "
                    f"mask before isolation -- see extract_spectrum.py digitize() docstring for why)"
                    if exclude_boxes else "")
                 + f". Diagnostics: {digitized['diagnostics']}",
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
