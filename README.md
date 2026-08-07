# Chinese Drug Label AE Extractor

Extracts adverse reaction (AE) mentions from a Chinese drug package insert
(说明书 / LPI) and codes each one against the real UMLS/MedDRA terminology —
with CUIs, not free text — so the output is usable for pharmacovigilance
data work instead of just a list of strings.

Demonstrated on Sanofi's Dupixent (dupilumab) Chinese label, but nothing in
the pipeline is Dupixent-specific — point it at any Chinese drug label PDF.

## Why this exists

A drug label's adverse reactions aren't confined to the "不良反应" section —
they're scattered through warnings, precautions, and population-specific
subsections too (see [LPI/dupixent_ae_reference.csv](LPI/dupixent_ae_reference.csv):
16 of the 37 verified terms came from outside the main AE table). Finding
them by hand doesn't scale past one document, and running them past a
translation model to match against English-language UMLS data turned out to
have a nasty failure mode — see below.

## Pipeline

```
extract_sections.py    PDF -> clean, section-tagged text
        |               (layout-aware: distinguishes real line breaks from
        |                mid-word PDF page-width wrapping — see the module
        |                docstring for why this matters)
        v
llm_extract.py          LLM reads each section in context, proposes AE
        |                mentions + a standard English MedDRA-style term
        |                for each (needs your own API key — see Setup)
        v
build_umls_index.py     Parses your local UMLS MRCONSO.RRF/MRSTY.RRF into
        |                a lookup table (run once, cached)
        v
verify_and_report.py    Every LLM-proposed term is checked against the real
                         UMLS data by EXACT STRING LOOKUP — not a similarity
                         score (see "The finding" below for why). A term
                         either matches a real MedDRA string or it's flagged
                         for manual coding, with embedding-based suggestions
                         attached only as an advisory hint for the reviewer.
```

A small human-reviewed dictionary (`LPI/ae_term_dictionary.csv`) sits in
front of the LLM call: once a term has been seen and confirmed, it's an
instant free lookup on every future run, so the LLM is only ever spent on
genuinely new mentions.

## The finding that shaped this design

The first version of this pipeline translated Chinese candidate phrases with
a local NMT model (`Helsinki-NLP/opus-mt-zh-en`) and matched the English
output against UMLS by cosine similarity — the standard "vectorize and
threshold" recipe. Calibrating it against the real UMLS/MedDRA data (not a
toy example — the actual 60k-term MedDRA subset) surfaced a specific,
reproducible problem:

| Chinese term | Mistranslation | Matched to (real MedDRA) | Cosine score |
|---|---|---|---|
| 结膜炎 (conjunctivitis) | "Meningitis" | **Meningitis** | 1.000 |
| 角膜炎 (keratitis) | "Cornelitis" (not a word) | **Cornelia de Lange syndrome** | 0.931 |
| 心血管血栓栓塞事件 | "...throttle embolism" | **Heart throbbing** | 0.920 |

All three score above where a normal "high confidence, auto-accept"
threshold would sit (e.g. 0.85). The embedding model measures *"does this
English text resemble a MedDRA term"* — not *"was this translated
correctly"* — so once a translation error produces plausible-sounding
English, the resulting mismatch is invisible to the score. A threshold
can't fix this: the score distribution for wrong matches overlaps almost
entirely with the distribution for correct ones. Full writeup and the
scripts themselves are kept in
[`scripts/legacy_nmt_embedding_approach/`](scripts/legacy_nmt_embedding_approach/README.md).

The fix wasn't a better translation model (a larger one, NLLB-200, was
tested and was no better — see the same writeup). It was replacing
**generative translation + similarity scoring** with **LLM term proposal +
deterministic lookup**: an LLM reading the source sentence in context
proposes a term, and that term is either an exact (or spelling-normalized)
match in the real MedDRA data or it isn't — there's no continuous score for
a wrong answer to hide behind.

## Setup

1. **UMLS account** (free): register at the
   [NLM UTS](https://www.nlm.nih.gov/research/umls/index.html), download the
   Metathesaurus, and place `MRCONSO.RRF` (required) and `MRSTY.RRF`
   (recommended — used to filter to AE-relevant semantic types) under
   `umls/`. These files are large (MRCONSO.RRF is ~2GB) and licensed —
   `.gitignore` already excludes them, don't commit them.

2. **LLM API key** for whichever provider you want (`scripts/llm_extract.py`
   supports Anthropic and OpenAI out of the box; anything with an
   OpenAI-compatible endpoint is a small change to `call_openai`):
   ```bash
   cp .env.example .env
   # fill in LLM_PROVIDER and the matching API key
   ```

3. **Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

```bash
python scripts/run_pipeline.py
```

Or run each stage individually (useful if you want to inspect intermediate
output, or you already have a cached `umls_meddra_index.parquet` and want to
skip rebuilding it):

```bash
python scripts/extract_sections.py LPI/your-label.pdf output/sections.json
python scripts/llm_extract.py output/sections.json output/llm_candidates.json
python scripts/build_umls_index.py umls output/umls_meddra_index.parquet
python scripts/verify_and_report.py output/llm_candidates.json output/umls_meddra_index.parquet output/ae_report.csv
```

Final report columns: Section · Pages · Chinese Term · Context (source
sentence, for provenance — kept in Chinese, since it's a verbatim quote used
to trace a result back to the source PDF) · Standard English Term · CUI ·
SAB · TTY · Status (dictionary hit / UMLS-verified / needs manual coding) ·
LLM Confidence · Note · Suggested Alternatives (advisory nearest-neighbor
suggestions, unverified terms only).

## Reference extraction

[LPI/dupixent_ae_reference.csv](LPI/dupixent_ae_reference.csv) — 37 AE terms
from Dupixent's Chinese label (300mg PFS formulation, 2025-07-11 revision),
each carrying a MedDRA CUI looked up in the real UMLS data. 21 come from the
label's own AE table; the other 16 are scattered through "特定不良反应描述"
and "注意事项" subsections — e.g. *eczema herpeticum* (疱疹性湿疹),
*eosinophilic granulomatosis with polyangiitis* (嗜酸性肉芽肿性多血管炎),
and the two COPD-specific injection-site terms that don't appear in the main
table at all.

**This is a hand-verified reference set, not pipeline output** — it was read
and coded manually, and its columns differ from what
`verify_and_report.py` produces. It serves two purposes: it seeds
`LPI/ae_term_dictionary.csv` (so those terms are free dictionary hits on
every future run), and it's the ground truth to check a pipeline run against.
The pipeline writes to `output/`, which is gitignored — so running it can
never overwrite the reviewed data.

## Limitations

- **Not a regulatory or clinical tool.** This is a research/portfolio
  project. Output needs pharmacovigilance-qualified human review before any
  real-world use — the "Status" and "LLM Confidence" columns are there to
  make that review fast, not to replace it.
- LLM term proposals are only as good as the model reading them; the
  deterministic UMLS check catches "doesn't exist in MedDRA" but not "exists
  in MedDRA, but isn't quite the right nuance" (see the `面部皮疹` → generic
  `Rash` and `心血管死亡` → `Cardiac death` rows in the reference extraction —
  both flagged medium-confidence for exactly this reason).
- **The LLM extraction stage has not been run end-to-end yet.**
  `extract_sections.py`, `build_umls_index.py`, and `verify_and_report.py`
  are exercised and working; `llm_extract.py` is written but untested, as it
  needs an API key.
- Tested on one Chinese-language biologic label. Labels with heavier tabular
  layouts, non-standard section headers, or scanned/non-text-layer PDFs will
  need adjustments to `extract_sections.py`'s heading-detection regex or an
  OCR pre-pass.
