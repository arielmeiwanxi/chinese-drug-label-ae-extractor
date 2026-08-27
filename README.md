# Drug Label Adverse Reaction Reconciliation

Compares the adverse reactions listed in two versions of a drug's labelling and
reports where they diverge, with every term normalised to a MedDRA Preferred
Term so that the comparison is meaningful rather than a string diff.

Demonstrated on dupilumab (Dupixent): the EU Summary of Product Characteristics
against the Chinese NMPA-registered label. Both are public documents.

## Why

In pharmacovigilance, a drug's local label has to be kept consistent with its
global reference document, and every version update means checking again. The
comparison itself is easy. Getting the two lists into a comparable state is not:

- **The two documents sit at different granularity.** Local labels list
  reactions at Lowest Level Term (LLT) level, reference documents at Preferred
  Term (PT) level. Historically this meant opening the MedDRA dictionary and
  looking each term up by hand to find its roll-up.
- **String matching does not work.** "Corneal ulcer" and "Ulcerative keratitis"
  are the same concept written two ways. Exact matching reports them as a
  discrepancy; on the test pair that is 2 of 21 table terms, a 9.5% false
  positive rate on that section alone.
- **Reactions are not confined to the adverse reactions section.** 16 of the 37
  terms in the Chinese label sit outside its main AE table, in warnings and
  population-specific subsections.
- **The reference document may be confidential**, which rules out sending it to
  an external AI service.

## How it works

```
build_meddra_index.py    UMLS MRCONSO/MRSTY -> MedDRA term index
                         (111,384 terms; 27,163 PT / 81,143 LLT)
        |
        +--> extract_smpc_ae.py     reference label PDF -> AE list   [no LLM]
        |
        +--> extract_sections.py    local label PDF -> section text
        |    llm_extract.py         section text -> AE list          [LLM]
        |    verify_and_report.py   verify each term against MedDRA
        |
        v
compare_labels.py        both lists -> reconciled diff at PT level
```

**The LLT to PT roll-up is a lookup, not a matching problem.** On a MedDRA LLT
row, UMLS holds the code of that term's Preferred Term in the SDUI column. So
the roll-up every term needs is a dictionary lookup: 81,143 of 81,143 LLTs
resolve, deterministically, offline. This is also what makes synonyms free.
"Fever" is an LLT whose PT is "Pyrexia"; terms that look nothing alike
normalise to the same concept without any fuzzy matching.

**The reference side needs no LLM.** Its terms are already English and already
MedDRA-shaped, so the whole document can be handled by dictionary lookup
against the index. It runs offline at zero cost, which is what makes the
approach usable when the reference document cannot leave the machine.

## The negative result

The first version of this project translated the Chinese terms into English
with a local NMT model and matched them against MedDRA by embedding
similarity, accepting anything above a confidence threshold. Calibrated against
59,479 real MedDRA terms, it failed in a way no threshold can fix:

| Chinese term | Mistranslated as | Matched to | Cosine |
|---|---|---|---|
| 结膜炎 (conjunctivitis) | "Meningitis" | **Meningitis** | 1.000 |
| 角膜炎 (keratitis) | "Cornelitis" (not a word) | **Cornelia de Lange syndrome** | 0.931 |
| 心血管血栓栓塞 (cardiovascular thromboembolism) | "...throttle embolism" | **Heart throbbing** | 0.920 |

All three score above where a "high confidence, auto-accept" line would sit.
The score measures whether the English text resembles a MedDRA term, not
whether the translation was correct, so once a mistranslation produces
plausible English the error is invisible to it. A larger translation model
(NLLB-200) was no better.

The fix was not a better model. It was replacing generative translation plus
similarity scoring with **deterministic lookup**: a term either resolves to a
real MedDRA concept or it is flagged for manual coding, with no continuous
score for a wrong answer to hide behind. Full write-up and the code are kept in
[`scripts/legacy_nmt_embedding_approach/`](scripts/legacy_nmt_embedding_approach/README.md).

## Results on the test pair

`compare_labels.py` on the EU SmPC vs the Chinese label:

| | Count |
|---|---|
| Present in both | 24 |
| Chinese label only | 13 |
| EU document only | 3 |
| Matched only after LLT to PT roll-up | 2 |
| Unresolved (not in MedDRA) | 1 |

The Chinese-only set clusters meaningfully: four cardiovascular thromboembolic
terms with no counterpart anywhere in the EU SmPC, four severe
eosinophil-related events, three COPD-specific injection site reactions, and
two terms from a China-specific clinical study.

Two coding-quality findings surfaced in the published EMA document itself:
`Facial rash` is coded at LLT level where a PT was expected (the correct PT,
`Rash`, is returned with the flag), and `Serum sickness reaction` is not a
valid MedDRA term.

Free-text extraction precision went from 17 false positives to 1 across four
fixes: claiming every occurrence of a matched term rather than only the first,
excluding the drug's own indications (read from the document's Section 4.1
rather than hardcoded), filtering by UMLS semantic type, and treating the
narrative subheadings as a separate higher-confidence tier.

## Setup

1. **UMLS account** (free): register at the
   [NLM UTS](https://www.nlm.nih.gov/research/umls/index.html), download the
   Metathesaurus, and place `MRCONSO.RRF` and `MRSTY.RRF` under `umls/`. These
   are large and licensed, and are gitignored. So is everything derived from
   them, including the generated index.
2. `pip install -r requirements.txt`
3. Only for the local-label narrative extraction: `cp .env.example .env` and
   add an API key.

## Usage

```bash
python scripts/build_meddra_index.py umls output/meddra_index.parquet
python scripts/extract_smpc_ae.py SmPC/your-smpc.pdf output/meddra_index.parquet output/smpc_ae.csv
python scripts/compare_labels.py output/smpc_ae.csv LPI/local_ae.csv output/meddra_index.parquet output/label_diff.csv
```

`compare_labels.py` auto-detects column names, so either side can come from an
extractor or from a hand-curated list.

## Reference extraction

[LPI/dupixent_ae_reference.csv](LPI/dupixent_ae_reference.csv) holds 37 AE terms
read out of the Chinese label by hand, each with a CUI verified against the real
UMLS data. It is **not pipeline output**: it seeds the dictionary so those terms
are free lookups on later runs, and it is the ground truth a pipeline run is
checked against.

## Limitations

- **Not a regulatory or clinical tool.** Output needs pharmacovigilance
  qualified review before any real-world use. Medical review is deliberately
  unchanged by this project; what changes is that the reviewer receives a
  traceable diff rather than a hand-compiled list.
- **The Chinese narrative extraction is not yet automated.** Tabulated sections
  run end to end. The free-text sections of the Chinese label were read
  manually for the reference set; `llm_extract.py` is written for this but has
  not been run end to end, as it needs an API key.
- Deterministic lookup catches "not in MedDRA" but not "in MedDRA, but not
  quite the right nuance" (see the `面部皮疹` and `心血管死亡` rows in the
  reference extraction, both flagged medium confidence for that reason).
- Tested on one document pair. Labels with heavier tabular layouts, different
  section headings, or scanned PDFs will need adjustments to the extractors.
