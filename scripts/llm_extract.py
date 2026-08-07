"""Stage 3 (LLM-assisted): read each section and identify genuine adverse
reaction mentions, proposing a standard English MedDRA-style term for each.

This replaces the earlier local-NMT-translation + embedding-similarity
approach. Calibrating that approach against the real UMLS/MedDRA data showed
a fundamental problem: a mistranslation's embedding score is indistinguishable
from a correct translation's score (both often land at 0.90-1.00 cosine
similarity), so no threshold can safely separate "confidently correct" from
"confidently wrong". An LLM reading the source sentence in context and
reasoning about which standard term applies does not have that failure mode
in the same way, and its output is never trusted blindly either — every
proposed term is deterministically checked against the real UMLS MedDRA data
in verify_and_report.py before being reported as confirmed.

Requires an API key for whichever provider you choose:
  export LLM_PROVIDER=anthropic   # or: openai
  export ANTHROPIC_API_KEY=...    # or: OPENAI_API_KEY=...

Usage: python llm_extract.py <sections.json> <llm_candidates.json>
"""
import json
import os
import re
import sys
from pathlib import Path

# Load .env so the API key can live in a file instead of having to be
# re-exported in every new shell session. Falls back silently to whatever is
# already in the environment if python-dotenv isn't installed.
try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).parent.parent / ".env")
except ImportError:
    pass

SYSTEM_PROMPT = """You are a pharmacovigilance coder extracting adverse reaction (AE) \
mentions from a Chinese drug package insert (说明书) for MedDRA coding.

For the section of text given, identify every genuine adverse reaction / safety \
signal mention — a symptom, sign, diagnosis, or lab abnormality that could happen \
to a patient taking the drug.

Do NOT extract:
- The disease(s) the drug treats (indications) or study population descriptions \
(e.g. "特应性皮炎患者" as a population, not as an AE)
- Dosing, regimen, or administration instructions
- Pharmacokinetic parameters, biomarkers used to measure efficacy, or trial \
efficacy endpoints
- Study design terms (e.g. "双盲", "安慰剂对照")
- Cross-references to other sections

For each genuine AE mention, return:
- "zh": the exact Chinese term/phrase as it appears
- "en": the standard English MedDRA-style term (Preferred Term style — e.g. \
"Injection site erythema", not a loose paraphrase)
- "context": a short snippet of the surrounding sentence for provenance
- "confidence": "high" | "medium" | "low" — your own confidence that this is a \
genuine, correctly-termed AE (medium/low for composite endpoints, ambiguous \
wording, or cases where you're unsure of the exact standard term)
- "note": brief reasoning ONLY if confidence is medium/low, or if you split a \
composite mention (e.g. "心血管血栓栓塞事件" into its components) — otherwise empty string

If a mention is a composite/grouped safety outcome (e.g. "心血管血栓栓塞事件（心血管死亡、\
非致死性心肌梗死和非致死性卒中）"), decompose it into its individually codable components \
rather than returning the composite phrase as one term.

Return ONLY a JSON array of objects with keys zh, en, context, confidence, note. \
Return an empty array [] if the section has no genuine AE mentions (this is \
expected and correct for sections like dosing, PK, or efficacy trial results)."""


def call_anthropic(section_text, model="claude-opus-4-8"):
    import anthropic
    client = anthropic.Anthropic()
    resp = client.messages.create(
        model=model,
        max_tokens=16000,
        # Deciding whether a phrase is a genuine AE and which standard term
        # applies is exactly the kind of judgment that benefits from thinking.
        # On this model family thinking is OFF unless requested explicitly.
        thinking={"type": "adaptive"},
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": f"Section text:\n\n{section_text}"}],
    )
    return first_text_block(resp.content)


def first_text_block(content):
    """Return the text of the first text block.

    Not content[0] — with thinking enabled the first block is a thinking
    block, so indexing position 0 would return the wrong block (or raise on
    the missing .text attribute)."""
    for block in content:
        if block.type == "text":
            return block.text
    return ""


def call_openai(section_text, model="gpt-4o"):
    from openai import OpenAI
    client = OpenAI()
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Section text:\n\n{section_text}"},
        ],
    )
    return resp.choices[0].message.content


def extract_json_array(text):
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if not match:
        return []
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return []


def main():
    if len(sys.argv) != 3:
        print("Usage: python llm_extract.py <sections.json> <llm_candidates.json>")
        sys.exit(1)
    in_path, out_path = sys.argv[1], sys.argv[2]

    provider = os.environ.get("LLM_PROVIDER", "anthropic").lower()
    call_fn = {"anthropic": call_anthropic, "openai": call_openai}.get(provider)
    if call_fn is None:
        print(f"Unknown LLM_PROVIDER={provider!r}. Use 'anthropic' or 'openai'.")
        sys.exit(1)

    with open(in_path, encoding="utf-8") as f:
        sections = json.load(f)

    results = []
    for sec in sections:
        # A section with almost no text (e.g. a one-line heading with no
        # body) can't contain an AE mention; skip the API call.
        if len(sec["text"]) < 10:
            continue
        print(f"[{provider}] scanning 【{sec['section']}】 ({sec['start_page']}-{sec['end_page']}, {len(sec['text'])} chars)...")
        raw = call_fn(sec["text"])
        items = extract_json_array(raw)
        print(f"  -> {len(items)} candidate AE mention(s)")
        for item in items:
            item["section"] = sec["section"]
            item["start_page"] = sec["start_page"]
            item["end_page"] = sec["end_page"]
            results.append(item)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n{len(results)} total candidate AE mentions -> {out_path}")


if __name__ == "__main__":
    main()
