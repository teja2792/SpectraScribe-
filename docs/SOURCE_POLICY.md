# Source policy — what SpectraScribe is allowed to mine, and why

This is the same underlying legal question SpectraVault's own
`docs/SOURCE_POLICY.md` had to answer, asked again here because
SpectraScribe's risk profile is different: SpectraVault mostly *downloads*
data that a database has already agreed to distribute in bulk (RRUFF, ROD,
COD). SpectraScribe *extracts* data out of a paper's own figures and
tables — a fundamentally more direct use of someone else's published
work, and one that needs its own explicit rule rather than inheriting
SpectraVault's reasoning by assumption.

## 1. Scope, for now: open-access papers only

SpectraScribe only ingests papers that are legitimately open access —
meaning the publisher has granted a license (CC-BY, CC-BY-NC, CC0, or
equivalent) allowing reuse of the paper's content, not just "the PDF
happens to be downloadable without a paywall prompt." A paper being
free-to-read is not the same as a paper being free-to-reuse; some
publishers post PDFs openly under a "free access" model while retaining
full copyright with no reuse rights at all. Every paper ingested must have
a machine-checkable, recorded open-access license before any figure or
table from it enters this pipeline.

This is a narrower rule than SpectraVault needed, on purpose: SpectraVault
distributes numeric facts extracted from copyrighted figures, which
`docs/SOURCE_POLICY.md` argues is defensible even from a paywalled source
(facts aren't copyrightable, only the particular expression is) — but that
argument gets weaker, not stronger, when the *extraction itself* is being
built as a general-purpose, repeatable pipeline rather than one curator
manually transcribing one number. Starting open-access-only avoids
resting an entire pipeline's legitimacy on a legal argument that's
genuinely more contestable at pipeline scale than at one-off scale.

## 2. What gets recorded, and why it's mandatory, not optional

Every record SpectraScribe produces must carry:

- **DOI** — the canonical, stable identifier for the source paper.
- **License** — the specific open-access license (CC-BY 4.0, CC0, etc.),
  not just a boolean "open access: yes."
- **Citation** — full bibliographic citation, generated from the paper's
  own metadata, not hand-typed.
- **Source location** — which figure or table number, and which panel if
  the figure is multi-panel (e.g. "Figure 3b," "Table 2"). Without this,
  "where did this number come from" stops being independently checkable
  against the original paper, which defeats the entire point of
  citing a source at all.

These are required fields in `schema.py`, not optional metadata a curator
might fill in later — a record missing any of them fails validation and is
never written.

## 3. What SpectraScribe does not do

- Does not ingest paywalled papers, even ones a user has personal
  institutional access to. Personal access to read a paper is not a
  license to feed it into an automated extraction pipeline that produces
  a redistributable dataset.
- Does not redistribute the original figure images or PDF pages
  themselves — only the extracted numeric data (positions, values) plus
  the citation metadata above. The original figure stays exactly where it
  already is: in the cited paper.
- Does not claim extracted data is equivalent evidence to a real
  instrument measurement. Every record's `source_type`/`extraction_method`
  (inherited from SpectraVault's confidence rubric — see `src/schema.py`)
  marks it as `digitized-from-figure`, scored lower than a bulk database
  download or an author-supplied raw file, on purpose.

## 4. Relationship to SpectraVault's policy

This document doesn't replace SpectraVault's `docs/SOURCE_POLICY.md` — it
answers the same question for a different, riskier kind of use. Anyone
reading both should come away with: SpectraVault reasons about
*redistributing facts already extracted by one curator*; this document
reasons about *building the extraction itself as repeatable, automated
infrastructure*, which is why the open-access-only line is drawn tighter
here.

Checked: 2026-07-28, by Ravi Teja AT, before any ingestion code was
written.
