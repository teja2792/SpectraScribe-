# Data layout

Every ingested paper gets exactly one folder, keyed by DOI, under `data/papers/`.
All of that paper's documents (main text, SI, corrections, ...) share this one
folder, one manifest, and one crops directory:

```
data/papers/<paper_id>/
    raw/
        ao5c08505.pdf
        ao5c08505_si_001.pdf
    crops/
        main_page002_Figure_1.png
        main_page003_Figure_2.png
        ...
        si_page002_Figure_S1.png
        si_page007_Table_S1.png
        ...
    manifest.json
```

`<paper_id>` is derived from the DOI by `paper_id_from_doi()` in `src/ingest_paper.py`
(registrant prefix stripped, suffix sanitized to a folder-safe string — e.g.
`10.1021/acsomega.5c08505` becomes `acsomega_5c08505`). One paper, one folder,
however many source PDFs it has.

## Why crop filenames are prefixed with `main_` / `si_`

A paper and its Supporting Information can both have a "Figure 1" — the SI's own
numbering usually gets an S-prefix ("Figure S1") in the *caption*, but that's a
convention, not a guarantee, and it doesn't help distinguish a main-text
`page002_Figure_1.png` from an SI `page002_Figure_1.png` if a paper ever breaks
that convention. Prefixing every crop filename with its doc_type (`main_`, `si_`)
guarantees no collision regardless of caption numbering, while still keeping
everything in one folder. Provenance also lives in the data itself: every region
in `manifest.json` carries a `"doc_type"` field, so "don't mix spectra together"
is enforced by keeping each record's origin attached to it, not by physically
separating documents into different folders.

## Re-running ingestion

`ingest()` refuses to overwrite an existing doc_type's regions unless called with
`--overwrite`. Re-parsing `main` only touches `main`-prefixed crops and `doc_type:
"main"` regions in the manifest; `si` data already in the same folder is left
alone. This is deliberate: silent overwrites were how an earlier version of this
pipeline accumulated `*_OLD`, `*_OLD2`, `*_OLD3`... debris folders during
iterative testing.

## What's tracked in git

Per `docs/SOURCE_POLICY.md` section 3, SpectraScribe does not redistribute source
PDFs or cropped figure/table images — only the extracted data and citation
metadata. Accordingly `.gitignore` excludes `raw/` and `crops/` under
`data/papers/*/`; `manifest.json` is tracked. Anyone cloning the repo gets the
metadata and can regenerate the crops themselves by re-running `ingest_paper.py`
against the DOI + a legally obtained copy of the PDF.

## History

An earlier version of this pipeline split each paper across `parsed/main/` and
`parsed/si/` subfolders (to guarantee no crop-filename collisions between
documents) before consolidating into the single-folder-with-prefixed-filenames
layout above. Before that, an even earlier version wrote flat, ungrouped output
under `data/raw_papers/`, with no per-paper folder at all and `*_OLD`-suffixed
debris left behind by failed overwrite attempts. Both are superseded; see the
corresponding chat response for the one-time cleanup commands.
