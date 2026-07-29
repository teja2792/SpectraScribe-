"""
fetch_metadata.py

Phase 2, part 1: turn a bare DOI into real citation metadata (title,
authors, journal, year, volume, pages) so that a record's "source_location"
(e.g. "Table S1") means something without a human going and looking up
the DOI by hand every time.

Two independent sources, tried in order, because neither is reliable
alone:

  1. CrossRef's free REST API (https://api.crossref.org/works/{doi}) --
     the correct long-term source: publisher-submitted, works for any
     DOI, no auth needed. NOT reachable from this project's current
     sandbox environment (network allowlist blocks api.crossref.org,
     same as it blocks the publisher sites directly -- confirmed via
     curl returning 403 blocked-by-allowlist, not a code bug). Kept as
     the first attempt because a normal (non-sandboxed) environment
     running this script will very likely reach it fine, and it's
     strictly more reliable than parsing PDF text when it's available.

  2. Best-effort extraction straight from the PDF's own first page,
     used automatically when CrossRef is unreachable. This is NOT a
     generic PDF-metadata parser -- it specifically targets the layout
     ACS Publications uses (title, then an author line, then a literal
     "Cite This: <Journal> <Year>, <Volume>, <Pages>" line -- see
     _CITE_THIS_RE), because that's the one real, verified layout this
     project has actually run against. A paper from a different
     publisher will very likely fail this heuristic; failure is
     reported explicitly (raises, doesn't guess), not silently wrong.

Whichever source succeeds, the result is written into the paper's
manifest.json under "citation_metadata" with "metadata_reviewed": false
-- this is the same checkpoint principle used for figures/tables
elsewhere in this project (see src/extract_table.py): an automated
first pass, never trusted as final until a human confirms it. Flip
"metadata_reviewed" to true by hand (or via --mark-reviewed) once you've
checked it against the actual paper.

Usage:
    python fetch_metadata.py --doi 10.xxxx/yyyy --pdf ../data/papers/<id>/raw/paper.pdf
    python fetch_metadata.py --doi 10.xxxx/yyyy --mark-reviewed
"""

import argparse
import json
import re
import urllib.request
import urllib.error
from datetime import datetime, timezone
from pathlib import Path

import fitz  # PyMuPDF

from ingest_paper import paper_id_from_doi


CROSSREF_TIMEOUT_S = 10

# ACS's own "Cite This:" line, e.g. "Cite This: ACS Omega 2026, 11, 5374-5383"
# -- printed by the publisher on every article's first page, right after the
# author list. Handles both a plain hyphen and the en-dash/minus-sign
# characters ACS actually typesets in page ranges (found by hitting it: the
# real PDF used U+2212 MINUS SIGN, not U+002D HYPHEN-MINUS, in "5374−5383" --
# a plain "-" in the regex silently matched zero times).
_CITE_THIS_RE = re.compile(
    r"Cite This:\s*(?P<journal>.+?)\s+(?P<year>\d{4}),\s*(?P<volume>\d+),\s*"
    r"(?P<pages>\d+[−\-–]\d+)",
)


def fetch_crossref_metadata(doi: str) -> dict | None:
    """Queries CrossRef's public API. Returns a metadata dict on success,
    or None on ANY failure (network unreachable, timeout, 404, malformed
    response) -- this is a best-effort lookup, not a required step, so it
    fails quietly and lets the caller fall back to PDF-text extraction
    rather than crashing the whole fetch_metadata() call over a network
    hiccup."""
    url = f"https://api.crossref.org/works/{doi}"
    req = urllib.request.Request(url, headers={"User-Agent": "SpectraScribe/0.1 (mailto:teja2792@gmail.com)"})
    try:
        with urllib.request.urlopen(req, timeout=CROSSREF_TIMEOUT_S) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, json.JSONDecodeError, OSError):
        return None

    msg = payload.get("message", {})
    if not msg:
        return None

    title = (msg.get("title") or [None])[0]
    authors = [
        f"{a.get('given', '').strip()} {a.get('family', '').strip()}".strip()
        for a in msg.get("author", [])
    ]
    journal = (msg.get("container-title") or [None])[0]
    year = None
    for date_key in ("published-print", "published-online", "published"):
        if date_key in msg and "date-parts" in msg[date_key]:
            parts = msg[date_key]["date-parts"][0]
            if parts:
                year = parts[0]
                break

    if not title or not authors:
        return None  # incomplete response -- don't hand back a half-record

    return {
        "title": title,
        "authors": authors,
        "journal": journal,
        "year": year,
        "volume": msg.get("volume"),
        "pages": msg.get("page"),
        "metadata_source": "CrossRef API",
    }


_AUTHOR_STOPWORDS = {
    "a", "an", "the", "of", "via", "for", "using", "in", "on", "with",
    "and", "to", "by", "from", "into", "as", "at", "supported", "based",
}


def _looks_like_author_line(line: str) -> bool:
    """True if a line is shaped like a comma-separated author list rather
    than a title fragment -- e.g. 'Petra Demeny, Borbala Tegze,* Balint
    Fodor,' vs 'Titania-Supported Photocatalytic Coatings of Cu2O
    Nanoparticles'. Needed because ACS wraps long author lists across
    multiple lines exactly the way it wraps long titles -- found by
    hitting it: assuming 'the author line' is always exactly the single
    line right before 'Cite This:' silently swallowed the real last
    author-list line into the title and left only the final few authors
    in the author field, on a real paper with an 11-author, two-line
    byline. Heuristic: split on commas/'and', strip footnote markers,
    and check whether most resulting tokens look like 2-4 capitalized
    words with no lowercase connector words -- title fragments are full
    of exactly those connector words ('of', 'via', 'for', ...), names
    aren't."""
    tokens = [t.strip(" *") for t in re.split(r",| and ", line) if t.strip(" *")]
    if not tokens:
        return False
    name_shaped = 0
    for tok in tokens:
        words = tok.split()
        if not (1 <= len(words) <= 4):
            continue
        if any(w.lower() in _AUTHOR_STOPWORDS for w in words):
            continue
        if not all(w[0].isupper() or not w[0].isalpha() for w in words):
            continue
        name_shaped += 1
    return name_shaped / len(tokens) > 0.5


def extract_metadata_from_pdf_text(pdf_path: str) -> dict:
    """Best-effort fallback when CrossRef is unreachable -- see module
    docstring for the exact layout this targets (ACS Publications'
    first-page format) and why it's not a generic solution. Raises
    ValueError (not a silent guess) if the expected "Cite This:" line
    isn't found, since that's the one anchor this heuristic actually
    trusts; everything else (title, authors) is positioned relative to
    it."""
    doc = fitz.open(pdf_path)
    page_text = doc[0].get_text()
    doc.close()

    lines = [ln.strip() for ln in page_text.split("\n") if ln.strip()]

    cite_match = _CITE_THIS_RE.search(page_text)
    if not cite_match:
        raise ValueError(
            f"Could not find a 'Cite This: <Journal> <Year>, <Volume>, <Pages>' line "
            f"in the first page of {pdf_path} -- this fallback only handles ACS "
            f"Publications' layout (see module docstring). Enter metadata by hand "
            f"in the manifest instead."
        )

    # Walk backward from "Cite This:", collecting consecutive author-shaped
    # lines (the byline can wrap across multiple lines, same as titles --
    # see _looks_like_author_line docstring for the real bug this fixes).
    # Whatever's left above that is the title, which can itself wrap
    # across multiple lines.
    cite_this_idx = next((i for i, ln in enumerate(lines) if ln.startswith("Cite This:")), None)
    if cite_this_idx is None or cite_this_idx == 0:
        raise ValueError(f"'Cite This:' line found but couldn't locate the preceding author line in {pdf_path}")

    author_lines = []
    i = cite_this_idx - 1
    while i >= 0 and _looks_like_author_line(lines[i]):
        author_lines.insert(0, lines[i])
        i -= 1
    if not author_lines:
        raise ValueError(f"Found 'Cite This:' in {pdf_path} but no author-shaped line before it -- layout may differ from expected ACS format.")

    title_lines = lines[:i + 1]
    title = " ".join(title_lines).strip()

    # Author lines look like "First Last, First Last,* First Last, and First Last,"
    # (wrapped across lines) -- strip corresponding-author markers (*) and
    # footnote symbols before splitting.
    cleaned = " ".join(author_lines).replace("*", "").replace(" and ", ", ")
    authors = [a.strip() for a in cleaned.split(",") if a.strip()]

    return {
        "title": title,
        "authors": authors,
        "journal": cite_match.group("journal").strip(),
        "year": int(cite_match.group("year")),
        "volume": cite_match.group("volume"),
        "pages": cite_match.group("pages").replace("−", "-").replace("–", "-"),
        "metadata_source": "extracted from PDF first page (ACS 'Cite This:' layout)",
    }


def build_citation_string(meta: dict, doi: str) -> str:
    """Plain natural-order author list (as printed), not reformatted into
    a specific citation style -- reformatting names into 'Last, F.' style
    automatically is exactly the kind of finicky, silently-wrong-prone
    transform this project avoids elsewhere (multi-word surnames,
    diacritics, suffixes). Downstream tooling can reformat from these
    fields for whatever citation style it needs; this string is a
    readable default, not a claim of matching any particular journal's
    format."""
    authors = ", ".join(meta["authors"])
    parts = [authors + "." if authors else "", meta.get("title", "") + "."]
    if meta.get("journal"):
        tail = meta["journal"]
        if meta.get("year"):
            tail += f" {meta['year']}"
        if meta.get("volume"):
            tail += f", {meta['volume']}"
        if meta.get("pages"):
            tail += f", {meta['pages']}"
        parts.append(tail + ".")
    parts.append(f"https://doi.org/{doi}")
    return " ".join(p for p in parts if p)


def fetch_metadata(doi: str, pdf_path: str, data_root: str, overwrite: bool = False) -> dict:
    paper_id = paper_id_from_doi(doi)
    manifest_path = Path(data_root) / "papers" / paper_id / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"{manifest_path} doesn't exist -- run ingest_paper.py for this paper first."
        )

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    if "citation_metadata" in manifest and not overwrite:
        raise FileExistsError(
            f"{manifest_path} already has citation_metadata -- pass --overwrite to redo it."
        )

    meta = fetch_crossref_metadata(doi)
    if meta is None:
        if not pdf_path:
            raise RuntimeError(
                "CrossRef lookup failed (unreachable or no match) and no --pdf was given "
                "to fall back to PDF-text extraction."
            )
        meta = extract_metadata_from_pdf_text(pdf_path)

    meta["citation"] = build_citation_string(meta, doi)
    meta["fetched_at_utc"] = datetime.now(timezone.utc).isoformat()
    meta["metadata_reviewed"] = False  # checkpoint: never trust an auto-fetch as final

    manifest["citation_metadata"] = meta
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    return manifest


def mark_reviewed(doi: str, data_root: str) -> None:
    """Flip citation_metadata.metadata_reviewed to true -- the human
    confirmation step. Separate, explicit call rather than a flag on
    fetch_metadata() so that 'fetched' and 'confirmed correct by a human'
    can never be accidentally conflated into one automatic step."""
    paper_id = paper_id_from_doi(doi)
    manifest_path = Path(data_root) / "papers" / paper_id / "manifest.json"
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    if "citation_metadata" not in manifest:
        raise FileNotFoundError(f"{manifest_path} has no citation_metadata to mark reviewed -- run fetch_metadata first.")
    manifest["citation_metadata"]["metadata_reviewed"] = True
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Phase 2: fetch citation metadata for a paper.")
    parser.add_argument("--doi", required=True)
    parser.add_argument("--pdf", help="Path to the main PDF, used as a fallback if CrossRef is unreachable.")
    parser.add_argument("--data-root", default="../data")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--mark-reviewed", action="store_true", help="Confirm existing citation_metadata as human-checked, instead of fetching.")
    args = parser.parse_args()

    if args.mark_reviewed:
        mark_reviewed(args.doi, args.data_root)
        print(f"Marked citation_metadata as reviewed for {args.doi}")
        return

    try:
        manifest = fetch_metadata(args.doi, args.pdf, args.data_root, args.overwrite)
    except (FileExistsError, FileNotFoundError, ValueError, RuntimeError) as e:
        print(f"[ERROR] {e}")
        raise SystemExit(1)

    meta = manifest["citation_metadata"]
    print(f"Metadata source: {meta['metadata_source']}")
    print(f"Title: {meta['title']}")
    print(f"Authors ({len(meta['authors'])}): {', '.join(meta['authors'])}")
    print(f"Journal: {meta.get('journal')} {meta.get('year')}, {meta.get('volume')}, {meta.get('pages')}")
    print(f"Citation: {meta['citation']}")
    print(f"[NOT YET REVIEWED] -- check this against the actual paper, then run "
          f"'python fetch_metadata.py --doi {args.doi} --mark-reviewed'")


if __name__ == "__main__":
    main()
