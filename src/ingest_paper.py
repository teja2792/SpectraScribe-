"""
ingest_paper.py

Phase 1 of SpectraScribe: turn a PDF into structured layout data --
per-page text, and detected figure/table regions with their captions and
a cropped image of each region -- without yet trying to read numbers out
of them. That's Phase 2 (tables) and Phase 3 (curves) on purpose; this
module's only job is "find the pieces," not "understand the pieces."

Approach, and why it's built this way:

  Structured sources (PMC XML, arXiv HTML/LaTeX) are the better long-term
  path when they exist -- their layout is already machine-readable, which
  sidesteps most of what makes raw PDF parsing hard (multi-column reflow,
  figures split across a page break). That importer isn't built yet
  (tracked as a follow-up, see NOTES at the bottom); this module handles
  the fallback case -- a plain PDF -- since that's what's actually
  available for most open-access papers on first contact.

  Figure/table detection here is a documented heuristic, not a claim of
  robustness: PyMuPDF gives real, reliable primitives (per-block text with
  bounding boxes, embedded image objects, page rasterization at any DPI),
  but there is no ground truth in a raw PDF for "this rectangular region is
  Figure 3." The heuristic used: find caption text blocks matching
  /^(Figure|Fig\.?|Table)\s+\d+/, then take the vertical band between the
  previous block and the caption (same page) as the figure/table's region,
  and rasterize that band. This works well for single-figure-per-page
  layouts (common in supplementary PDFs and many single-column journal
  formats) and less well for dense multi-figure grids on one page --
  logged explicitly as a known failure mode (see detect_regions()
  docstring), not silently assumed away.

Usage:
    python ingest_paper.py --pdf path/to/paper.pdf --doi 10.xxxx/yyyy \
        --license CC-BY-4.0 --doc-type main
    python ingest_paper.py --pdf path/to/paper_si.pdf --doi 10.xxxx/yyyy \
        --license CC-BY-4.0 --doc-type si
    # Both land in the same data/papers/<paper_id>/ folder -- run once per
    # document (main, si, ...), not once per paper.
"""

import argparse
import hashlib
import json
import re
from pathlib import Path

import fitz  # PyMuPDF


CAPTION_RE = re.compile(r"^(Figure|Fig\.?|Table)\s+(S?\d+)[.:]\s*(.*)", re.IGNORECASE | re.DOTALL)
# S-prefix (e.g. "Figure S2", "Table S1") is standard numbering for
# Supporting Information documents -- found only after running against a
# real paper's SI PDF, where the original digit-only pattern silently
# matched zero regions in an 11-page document that actually contained at
# least 8 figures and a table (including several real spectra: a Raman
# spectrum of the bare glass substrate, a second Raman spectrum, and two
# XPS spectra). A pattern that "worked" on the main text but silently
# found nothing in the SI would have been the most dangerous kind of bug
# here -- not a crash, just quietly missing data.
#
# The [.:] after the number is REQUIRED, not optional -- also found by
# hitting it: an in-text body paragraph beginning "Figure S1 illustrates
# the experimental steps..." starts its own block with the exact same
# words as the real caption "Figure S1. Sample preparation: ..." but has
# no punctuation immediately after the number, since it's mid-sentence.
# Making the period/colon mandatory is what distinguishes a genuine
# caption from an in-text reference that happens to open a paragraph --
# without it, both matched, produced two regions with the identical
# label, and collided on the same output filename.
DOI_RE = re.compile(r"\b10\.\d{4,9}/[^\s\"<>]+")


SIDEBAR_MAX_WIDTH_PTS = 15  # narrower than any real column of body text


def extract_page_text_blocks(page: "fitz.Page") -> list:
    """Returns a list of {bbox, text} for every real body-text block on
    the page. A 'block' here is PyMuPDF's own paragraph-level grouping,
    not a single line -- good enough to find caption starts without
    needing a custom layout model.

    Filters out narrow rotated sidebar strips (found on the first real
    test PDF: ACS stamps a vertical 'Downloaded from
    http://pubs.acs.org/...' watermark down the page margin, rendered as
    a single text block spanning nearly the full page HEIGHT but only ~8
    points of WIDTH). Before this filter, that one block's bbox
    (y0=198.9, y1=602.1) was being picked up as "the previous block" for
    multiple real captions purely because its y-range overlapped theirs,
    corrupting every region on the page even after fixing the sort-order
    bug -- an aspect-ratio check (width < SIDEBAR_MAX_WIDTH_PTS) catches
    this class of watermark without needing to hardcode its text."""
    blocks = []
    for b in page.get_text("blocks"):
        x0, y0, x1, y1, text, block_no, block_type = b
        if block_type != 0:  # 0 = text block; skip image blocks here
            continue
        text = text.strip()
        if not text:
            continue
        if (x1 - x0) < SIDEBAR_MAX_WIDTH_PTS:
            continue  # rotated sidebar/watermark strip, not body content
        blocks.append({"bbox": (x0, y0, x1, y1), "text": text})
    return blocks


def find_doi(full_text: str) -> str | None:
    """Best-effort DOI extraction from the paper's own text -- checked by
    a human against the actual paper before a record is trusted (see
    docs/SOURCE_POLICY.md); this is a starting point; not authoritative on
    its own since the same regex can match a DOI mentioned in a citation
    to a *different* paper (e.g. in the references list) rather than the
    paper's own DOI."""
    match = DOI_RE.search(full_text)
    return match.group(0).rstrip(").,;") if match else None


CAPTION_MERGE_GAP_PTS = 8   # gap below which a following block is treated as
                            # a line-wrapped continuation of the same caption,
                            # not a separate piece of content
MIN_BOUNDARY_BLOCK_WIDTH = 100  # narrower blocks are treated as axis-label
                                 # fragments, not real paragraph boundaries


def _looks_like_axis_label(block: dict) -> bool:
    """True if a block is more likely a chart's own tick label / axis
    title than real surrounding body text -- found necessary after a real
    XPS spectrum figure's region got clipped to a 17pt sliver because the
    plot's own 'Binding energy (eV)' axis label and a row of tick numbers
    (extractable as real text, since the chart was vector-drawn rather
    than a flattened raster image) were mistaken for the nearest body
    paragraph. Two independent signals, either one is enough to exclude a
    block from counting as a region boundary: it's narrower than
    MIN_BOUNDARY_BLOCK_WIDTH (tick labels and short axis titles are much
    narrower than a real paragraph column), or more than half its
    non-whitespace characters are digits (a row of axis tick values reads
    almost entirely as numbers; body prose doesn't)."""
    x0, y0, x1, y1 = block["bbox"]
    if (x1 - x0) < MIN_BOUNDARY_BLOCK_WIDTH:
        return True
    stripped = block["text"].replace(" ", "").replace("\n", "").replace("|", "")
    if stripped and sum(c.isdigit() for c in stripped) / len(stripped) > 0.5:
        return True
    return False


def get_page_image_bboxes(page: "fitz.Page") -> list:
    """Bounding boxes (in page coordinates) of every embedded raster image
    on the page. Used as ground truth for figures that are actual
    embedded photos (e.g. an SEM/AFM micrograph) -- for these, there is
    often almost no text-layer gap between the surrounding paragraphs and
    the caption (the image fills that visual space, but has no text
    'presence' at all), which silently defeats the text-block-gap
    heuristic in detect_regions(). Found on a real SEM figure whose
    caption sat only 9pt below the preceding section heading in the text
    layer -- the actual photo was invisible to a text-only view of the
    page entirely."""
    return [fitz.Rect(info["bbox"]) for info in page.get_image_info()]


def detect_regions(doc: "fitz.Document") -> list:
    """Finds figure/table regions across the whole document.

    Known failure modes, stated up front rather than discovered later:
      - Multi-panel figures on a dense page (e.g. 2x2 grid of subplots
        under one caption) are returned as ONE region -- panel-splitting
        is a documented follow-up (see NOTES), not handled here yet.
      - A caption appearing directly at the top of a page (content
        continuing from the previous page) will produce an empty or
        near-empty region, since there's no "previous block" on this page
        to bound it -- these are flagged with a zero-or-tiny bbox height
        rather than silently returning a bad crop.
      - Dense multi-figure pages (e.g. 4 separate figures stacked closely)
        rely on caption-to-caption spacing being large enough to
        distinguish one figure's content from the next; tightly packed
        layouts will over- or under-crop. Each region's bbox is recorded
        so a human reviewing the crop can immediately see if it's wrong.

    Fixed after hitting each on a real paper, not anticipated in advance:
      (1) PyMuPDF's get_text("blocks") order follows content-stream
      order, not visual position -- fixed by sorting by y0 per page.
      (2) Two-column body text with column-width AND full-width figures
      -- fixed by scoping the boundary search to the caption's own column
      unless the caption itself spans (near-)full page width.
      (3) Multi-line captions get split across two text blocks by
      PyMuPDF, and the wrapped continuation was being mistaken for real
      following content (clipped a table to <1pt tall) -- fixed by
      merging any immediately-following block within
      CAPTION_MERGE_GAP_PTS into the caption itself before searching for
      the true boundary.
      (4) Vector-drawn charts with real extractable axis-label text (not
      flattened into a raster image) had their own axis title/tick labels
      mistaken for a preceding paragraph -- fixed by excluding
      axis-label-shaped blocks (see _looks_like_axis_label()) from
      counting as boundary candidates.
      (5) A real embedded photo (SEM micrograph) sitting between two text
      blocks with almost no text-layer gap was invisible to the
      text-only heuristic entirely -- fixed by checking the page's actual
      embedded image bounding boxes (get_page_image_bboxes()) first, and
      only falling back to the text-block heuristic when no embedded
      image is positioned appropriately for this caption (true for
      vector-drawn plots, which have no embedded raster image at all).
    """
    regions = []
    for page_num, page in enumerate(doc):
        blocks = extract_page_text_blocks(page)
        image_bboxes = get_page_image_bboxes(page)
        page_width = page.rect.width
        page_height = page.rect.height
        page_mid_x = page_width / 2

        def column_of(bbox):
            center_x = (bbox[0] + bbox[2]) / 2
            return "left" if center_x < page_mid_x else "right"

        blocks_sorted = sorted(blocks, key=lambda b: b["bbox"][1])
        for idx, block in enumerate(blocks_sorted):
            m = CAPTION_RE.match(block["text"])
            if not m:
                continue

            label_kind = m.group(1).lower()
            label_num = m.group(2)
            caption_text = block["text"]
            cap_bbox = list(block["bbox"])

            # Merge immediately-following line-wrapped continuation
            # blocks into the caption (see docstring, fix (3)) before
            # doing anything else -- this must happen first, since it
            # changes where the caption's true bottom edge (cap_y1) is.
            look_idx = idx + 1
            while look_idx < len(blocks_sorted):
                nxt = blocks_sorted[look_idx]
                if CAPTION_RE.match(nxt["text"]):
                    break  # a real new caption, not a continuation
                if nxt["bbox"][1] - cap_bbox[3] > CAPTION_MERGE_GAP_PTS:
                    break  # real gap -- not a continuation
                caption_text += " " + nxt["text"]
                cap_bbox[3] = nxt["bbox"][3]
                cap_bbox[0] = min(cap_bbox[0], nxt["bbox"][0])
                cap_bbox[2] = max(cap_bbox[2], nxt["bbox"][2])
                look_idx += 1

            cap_x0, cap_y0, cap_x1, cap_y1 = cap_bbox
            is_full_width = (cap_x1 - cap_x0) > (page_width * 0.6)

            if is_full_width:
                scoped_blocks = blocks
            else:
                target_col = column_of(cap_bbox)
                scoped_blocks = [b for b in blocks if column_of(b["bbox"]) == target_col]

            is_table = label_kind.startswith("table")
            if is_table:
                # A table's own header row and data cells are SHORT,
                # numeric-heavy blocks -- exactly what _looks_like_axis_label
                # flags. That's the wrong filter to apply here: for a
                # figure we want to skip past such fragments to find the
                # true preceding paragraph, but for a table those
                # fragments ARE the content the region needs to include.
                # Found by hitting it: filtering them out left the
                # table's own header row as the nearest surviving
                # candidate, clipping the region to a few points between
                # the caption and its own header. Fixed by only treating
                # a block as a real "table has ended" boundary if it's
                # either a new caption or genuine prose (long text) --
                # short table-row-shaped blocks are walked through, not
                # stopped at.
                stop_candidates = [
                    b["bbox"][1] for b in scoped_blocks
                    if b["bbox"][1] >= cap_y1 and b["bbox"] != tuple(cap_bbox)
                    and (len(b["text"]) > 100 or CAPTION_RE.match(b["text"]))
                ]
                region_y0 = cap_y1
                region_y1 = min(stop_candidates) if stop_candidates else page_height
                region_x0, region_x1 = (0, page_width) if is_full_width else (cap_x0, cap_x1)
            else:
                # Prefer real embedded-image bounds over the text-gap
                # heuristic when an image is actually positioned above
                # this caption (fix (5)) -- catches real photos the text
                # layer can't see at all.
                candidate_images = [r for r in image_bboxes if r.y1 <= cap_y0 + 2
                                     and (cap_y0 - r.y1) < 80]
                if candidate_images:
                    best_image = max(candidate_images, key=lambda r: r.y1)  # closest above the caption
                    region_x0, region_y0 = best_image.x0, best_image.y0
                    region_x1, region_y1 = best_image.x1, cap_y0
                else:
                    fig_scoped = [b for b in scoped_blocks if not _looks_like_axis_label(b)]
                    above = [b["bbox"][3] for b in fig_scoped if b["bbox"][3] <= cap_y0 and b["bbox"] != tuple(cap_bbox)]
                    region_y0 = max(above) if above else 0
                    region_y1 = cap_y0
                    region_x0, region_x1 = (0, page_width) if is_full_width else (cap_x0, cap_x1)

            region_bbox = (region_x0, max(0, region_y0), region_x1, min(page_height, region_y1))
            region_height = region_bbox[3] - region_bbox[1]

            regions.append({
                "kind": "table" if is_table else "figure",
                "label": f"{m.group(1)} {label_num}",
                "caption": caption_text,
                "page_number": page_num + 1,  # 1-indexed, matches how papers are cited
                "region_bbox": region_bbox,
                "region_height_pts": region_height,
                "low_confidence": region_height < 20,  # near-zero-height crop, likely wrong
            })

    # Defense-in-depth dedup: even with the mandatory-punctuation fix
    # above, keep only one region per normalized label (e.g. "Figure S1").
    # Ties are broken by region_height_pts, NOT by first-seen order --
    # found necessary on a real page where the source PDF itself has two
    # genuinely different "Figure S6" captions (an authoring duplicate:
    # one in-text-reference-shaped sentence that happens to satisfy the
    # caption punctuation rule, and the real caption sitting right next
    # to the actual embedded image). "Keep first occurrence" silently
    # kept the wrong one; the real caption is reliably the one with a
    # genuine (larger) region next to it, since a spurious match has no
    # real content adjacent to it to bound a region from.
    best_by_label = {}
    for region in regions:
        key = region["label"].lower().replace(" ", "")
        if key not in best_by_label or region["region_height_pts"] > best_by_label[key]["region_height_pts"]:
            if key in best_by_label:
                dropped = best_by_label[key]
                print(f"  [WARN] duplicate label {region['label']!r}: keeping the page {region['page_number']} "
                      f"occurrence ({region['region_height_pts']:.0f}pt region) over page {dropped['page_number']} "
                      f"({dropped['region_height_pts']:.0f}pt region)")
            best_by_label[key] = region

    return list(best_by_label.values())


def crop_region(doc: "fitz.Document", region: dict, out_dir: Path, dpi: int = 300, filename_prefix: str = "") -> str:
    """Rasterizes one detected region to a PNG at the given DPI (300 is a
    reasonable default for later axis/tick-mark OCR in Phase 3 -- higher
    than screen resolution on purpose, since small tick labels need real
    pixel density to OCR reliably). Returns the saved file path as a
    string, or None if the region is degenerate (zero/negative area --
    e.g. a caption with no preceding block on its page, so region_y0 ==
    region_y1). Found by hitting it, not by anticipating it: the first
    real run against an actual paper crashed here with a bare MuPDF
    'Invalid bandwriter header dimensions' error on a zero-height rect --
    low_confidence flagged the region but didn't stop the crop attempt.
    Fixed by checking the rect's actual area before ever calling
    get_pixmap, and returning None (recorded on the region, not silently
    dropped) instead of trying to rasterize nothing."""
    clip = fitz.Rect(*region["region_bbox"])
    if clip.is_empty or clip.width <= 0 or clip.height <= 0:
        return None

    page = doc[region["page_number"] - 1]
    zoom = dpi / 72  # PDF points are 1/72 inch; fitz zoom is a multiplier on that
    matrix = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=matrix, clip=clip)

    safe_label = region["label"].replace(" ", "_").replace(".", "")
    out_path = out_dir / f"{filename_prefix}page{region['page_number']:03d}_{safe_label}.png"
    pix.save(str(out_path))
    return str(out_path)


def paper_id_from_doi(doi: str) -> str:
    """Turns a DOI into a filesystem-safe folder name, e.g.
    '10.1021/acsomega.5c08505' -> 'acsomega_5c08505'. Drops the '10.xxxx/'
    registrant prefix (constant per publisher, not useful for telling
    papers apart at a glance) and keeps the suffix, which is normally
    already a readable per-article code. Falls back to a hash of the
    full DOI if the suffix turns out empty or degenerate, so this never
    silently collapses two different papers onto the same folder."""
    suffix = doi.split("/", 1)[1] if "/" in doi else doi
    safe = re.sub(r"[^a-zA-Z0-9]+", "_", suffix).strip("_").lower()
    if not safe:
        safe = hashlib.sha256(doi.encode("utf-8")).hexdigest()[:12]
    return safe


def ingest(pdf_path: str, doi: str, license_str: str, doc_type: str,
           data_root: str, overwrite: bool = False) -> dict:
    """End-to-end Phase 1 entry point: parse a PDF, detect regions, crop
    each to an image, and write a manifest describing what was found.
    Does not attempt to read any numbers out of the crops -- that's
    Phase 2/3's job, working from this manifest's output.

    Output layout: one folder per paper, keyed by a filesystem-safe id
    derived from its DOI (paper_id_from_doi()) -- ALL of a paper's
    documents (main text, SI, ...) land in this same folder, sharing one
    manifest and one crops/ directory:

        data/papers/<paper_id>/
            raw/<original filename>.pdf   (one per document, moved in)
            crops/main_page002_Figure_1.png, si_page002_Figure_S1.png, ...
            manifest.json                 (one file, all documents' regions)

    An earlier version split main and si into separate parsed/main/ and
    parsed/si/ subfolders to guarantee no filename collisions between the
    two documents' figure numbering (both a paper and its SI can have a
    "Figure 1"). That turned "one folder per paper" into multiple folders
    per paper, which was the wrong tradeoff -- the same guarantee is
    achieved more simply by prefixing every crop filename with its
    doc_type (main_/si_), so this consolidates back into a single folder
    without losing the collision safety.

    Each region in the manifest carries its own "doc_type" field, so
    main-text and SI regions are still distinguishable in the data even
    though they now live side by side -- this is what "don't mix spectra
    together" actually requires (provenance stays attached to each
    record), not physical folder separation.

    Re-running a doc_type that's already in the manifest is a clean
    refusal unless --overwrite is passed; overwrite removes only that
    doc_type's own regions and crop files, leaving the other doc_type's
    data untouched. A failed deletion (e.g. a read-only-mount issue)
    raises a clear, actionable error instead of a bare traceback.
    """
    if doc_type not in ("main", "si"):
        raise ValueError(f"doc_type must be 'main' or 'si', got {doc_type!r}")

    paper_id = paper_id_from_doi(doi)
    paper_dir = Path(data_root) / "papers" / paper_id
    raw_dir = paper_dir / "raw"
    crops_dir = paper_dir / "crops"
    manifest_path = paper_dir / "manifest.json"

    existing_manifest = None
    if manifest_path.exists():
        with open(manifest_path, "r", encoding="utf-8") as f:
            existing_manifest = json.load(f)
        if doc_type in existing_manifest.get("documents", {}) and not overwrite:
            raise FileExistsError(
                f"{manifest_path} already has a {doc_type!r} document parsed -- "
                f"pass --overwrite to redo just this doc_type, or use a different --doc-type."
            )

    raw_dir.mkdir(parents=True, exist_ok=True)
    crops_dir.mkdir(parents=True, exist_ok=True)

    # If overwriting, remove this doc_type's old crop files before writing
    # new ones -- old regions for OTHER doc_types are left alone.
    if existing_manifest and doc_type in existing_manifest.get("documents", {}):
        stale_regions = [r for r in existing_manifest["regions"] if r["doc_type"] == doc_type]
        for r in stale_regions:
            if r.get("crop_path"):
                try:
                    Path(r["crop_path"]).unlink(missing_ok=True)
                except OSError as e:
                    raise OSError(
                        f"Could not remove stale crop {r['crop_path']} to re-parse "
                        f"doc_type={doc_type!r} ({e}). This happens on some network-mounted "
                        f"folders that don't allow deletion from this process -- delete files "
                        f"matching '{doc_type}_*' in {crops_dir} yourself and re-run."
                    ) from e

    source = Path(pdf_path)
    organized_pdf_path = raw_dir / source.name
    if not organized_pdf_path.exists():
        import shutil
        shutil.copy2(source, organized_pdf_path)
        # Copy, not move: the source may be sitting in a shared inbox
        # folder the user drops new papers into, and silently vanishing
        # a file the user just placed there would be a worse surprise
        # than the folder having one extra copy.

    doc = fitz.open(str(organized_pdf_path))
    full_text = "\n".join(page.get_text() for page in doc)

    detected_doi = doi or find_doi(full_text)
    regions = detect_regions(doc)

    for region in regions:
        region["doc_type"] = doc_type
        region["crop_path"] = crop_region(doc, region, crops_dir, filename_prefix=f"{doc_type}_")
        if region["crop_path"] is None:
            region["low_confidence"] = True
            region["crop_failed_reason"] = "degenerate region (zero/negative area) -- see caption-detection heuristic limitations in module docstring"

    doc_n_pages = doc.page_count
    doc.close()

    # Merge into the paper-level manifest: keep every other doc_type's
    # regions/metadata untouched, replace only this doc_type's.
    all_regions = [] if existing_manifest is None else [
        r for r in existing_manifest["regions"] if r["doc_type"] != doc_type
    ]
    all_regions.extend(regions)

    documents = {} if existing_manifest is None else dict(existing_manifest.get("documents", {}))
    documents[doc_type] = {
        "pdf_path": str(organized_pdf_path),
        "n_pages": doc_n_pages,
        "doi_source": "provided" if doi else ("auto-detected, unverified" if detected_doi else None),
    }

    manifest = {
        "paper_id": paper_id,
        "doi": detected_doi,
        "license": license_str,
        "documents": documents,
        "n_regions_detected": len(all_regions),
        "n_figures": sum(1 for r in all_regions if r["kind"] == "figure"),
        "n_tables": sum(1 for r in all_regions if r["kind"] == "table"),
        "n_low_confidence_regions": sum(1 for r in all_regions if r["low_confidence"]),
        "regions": all_regions,
    }

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    return manifest


def main():
    parser = argparse.ArgumentParser(description="Phase 1: ingest a PDF, detect figure/table regions.")
    parser.add_argument("--pdf", required=True, help="Path to the source PDF.")
    parser.add_argument("--doi", required=True, help="DOI -- required, used to derive the paper's folder name (see docs/SOURCE_POLICY.md: every record must be DOI-traceable).")
    parser.add_argument("--license", required=True, help="Open-access license, e.g. CC-BY-4.0 (see docs/SOURCE_POLICY.md).")
    parser.add_argument("--doc-type", default="main", choices=["main", "si"], help="Whether --pdf is the main article or its Supporting Information (default: main).")
    parser.add_argument("--data-root", default="../data", help="Root data/ folder (default: ../data, relative to src/).")
    parser.add_argument("--overwrite", action="store_true", help="Re-parse even if this paper/doc_type was already parsed.")
    args = parser.parse_args()

    try:
        manifest = ingest(args.pdf, args.doi, args.license, args.doc_type, args.data_root, args.overwrite)
    except (FileExistsError, OSError, ValueError) as e:
        # Clean one-line refusal for expected failure modes (already
        # parsed, can't overwrite, bad doc_type) instead of a raw
        # traceback -- those aren't bugs, they're this script correctly
        # declining to do something destructive or redundant.
        print(f"[ERROR] {e}")
        raise SystemExit(1)

    doc_info = manifest["documents"][args.doc_type]
    print(f"Parsed {doc_info['n_pages']} pages from {args.pdf} (doc_type={args.doc_type})")
    print(f"Organized under data/papers/{manifest['paper_id']}/")
    print(f"DOI: {manifest['doi']} ({doc_info['doi_source']})")
    n_this_doc = sum(1 for r in manifest["regions"] if r["doc_type"] == args.doc_type)
    print(f"Detected {n_this_doc} regions in this document "
          f"(paper total so far: {manifest['n_regions_detected']} regions, "
          f"{manifest['n_figures']} figures, {manifest['n_tables']} tables across "
          f"{len(manifest['documents'])} document(s))")
    if manifest["n_low_confidence_regions"]:
        print(f"  [WARN] {manifest['n_low_confidence_regions']} region(s) flagged low-confidence "
              f"(near-zero-height crop) -- check these by hand before trusting them")


if __name__ == "__main__":
    main()


# NOTES -- follow-ups deliberately not built in this first pass:
#   1. Structured-source importer (PMC XML / arXiv HTML) as a preferred
#      path over PDF+heuristic parsing, for papers where it's available.
#   2. Multi-panel figure splitting (a/b/c/d sub-panels under one caption
#      currently return as a single region).
#   3. A real caption-to-region association model (e.g. detecting actual
#      image/drawing object bounding boxes on the page and associating
#      them with the nearest caption by proximity) instead of the
#      previous-block-to-caption vertical-band heuristic used here --
#      would handle dense multi-figure pages correctly, which this
#      version documents as a known weak spot rather than silently
#      mishandling.
