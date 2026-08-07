# Legacy approach: local NMT + embedding similarity (kept for reference)

This folder holds the first version of the AE-extraction pipeline: rule-based
candidate phrase extraction (`prefilter.py`), local neural machine translation
of Chinese candidates to English (`translate.py`, using
`Helsinki-NLP/opus-mt-zh-en`), and cosine-similarity matching against a local
UMLS/MedDRA embedding index (`match_classify.py`, using
`pritamdeka/S-PubMedBert-MS-MARCO`).

**It's kept, not deleted, because the negative result is itself the
interesting finding.** Calibrating this approach against the real UMLS data
surfaced a failure mode worth documenting: mistranslated terms did not score
lower than correct ones.

- "结膜炎" (conjunctivitis) → mistranslated as "Meningitis" → matched
  "Meningitis" in MedDRA at **cosine similarity 1.000**
- "角膜炎" (keratitis) → mistranslated as "Cornelitis" (not a real word) →
  matched "Cornelia de Lange syndrome" (an unrelated genetic syndrome) at
  **0.931**
- "心血管血栓栓塞事件" → "血栓" (thrombosis) mistranslated as "throttle" →
  matched "Heart throbbing" at **0.920**

All of these sit well above where a "high confidence, auto-accept" threshold
would normally be drawn (e.g. 0.85). In other words: **the embedding score
measures "does this English text resemble a MedDRA term", not "was this
translated correctly"** — so it cannot be used to gate automatic acceptance.
A translation error and a correct translation are, to the embedding model,
often indistinguishable.

This is why the main pipeline (see the project README) replaced the
translate-then-embed approach with an LLM reading the source Chinese in
context and proposing a term directly, which is then checked against the
real MedDRA data via **exact lookup**, not similarity scoring — a term either
matches a real MedDRA string or it's flagged, with no ambiguous middle
ground for a bad guess to hide in.

These scripts still run standalone if you want to reproduce the comparison
yourself; they are no longer called by `run_pipeline.py`.
