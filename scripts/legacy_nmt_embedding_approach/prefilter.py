"""Stage 2: rule-based prefilter for AE-like candidate phrases.

Reads the section-tagged JSON from extract_sections.py, splits section text
into sentences, then further splits sentences into phrase-level candidates
(UMLS terms are short phrases, not full sentences, so matching whole
sentences against them would systematically depress similarity scores).

Usage: python prefilter.py <sections.json> <candidates.json>
"""
import json
import re
import sys

SENTENCE_SPLIT_RE = re.compile(r"[。！？；]")
# Split on list punctuation AND parentheses: AE terms are often embedded
# mid-sentence right before a parenthetical breakdown, e.g.
# "心血管血栓栓塞事件（心血管死亡、非致死性心肌梗死...）" — without splitting on
# the opening paren too, "事件（心血管死亡" would stay fused as one fragment.
# Also split on frequency-category words: table rows render as
# "感染及侵染类疾病 常见 结膜炎" with no punctuation between the SOC name,
# frequency, and term, so without this "常见" etc. never separates them.
# Longest-first so "十分常见"/"十分罕见" aren't cut short by "常见"/"罕见".
FREQ_WORDS = ["十分常见", "十分罕见", "常见", "偶见", "罕见", "不详"]
PHRASE_SPLIT_RE = re.compile(r"[、，,\n（）()]|" + "|".join(FREQ_WORDS))
# Trailing PDF table-footnote reference markers (*, †, #), possibly stacked
# ("*†", "*#"), are not part of the term itself.
FOOTNOTE_MARKER_RE = re.compile(r"[*†#]+$")
HAS_CHINESE_RE = re.compile(r"[一-鿿]")

# Deliberately generous / high-recall. Precision is handled downstream by
# the embedding similarity step, not here.
SIGNAL_KEYWORDS = [
    "不良反应", "不良事件", "发生率", "报告了", "报告", "观察到", "可能出现",
    "症状", "体征", "十分常见", "常见", "偶见", "罕见", "十分罕见", "不详",
    "过敏", "反应", "事件", "病例", "增多", "减少", "障碍", "疼痛", "皮疹",
    "水肿", "炎", "综合征", "感染", "血栓", "栓塞", "出血", "休克",
]

# Frequency-category words often prefix a list of terms; stripping them
# out as noise words when they appear as a standalone phrase fragment.
NOISE_PHRASES = {
    "十分常见", "常见", "偶见", "罕见", "十分罕见", "不详", "见【注意事项】",
    "见【不良反应】", "见【临床药理】", "见【临床试验】",
}


def looks_like_ae(sentence: str) -> bool:
    return any(kw in sentence for kw in SIGNAL_KEYWORDS)


def clean_phrase(phrase: str) -> str:
    phrase = phrase.strip()
    phrase = re.sub(r"^[（(]|[）)]$", "", phrase).strip()
    phrase = re.sub(r"[【】（）()]", "", phrase).strip()
    phrase = FOOTNOTE_MARKER_RE.sub("", phrase).strip()
    return phrase


def split_candidates(sentence: str):
    phrases = []
    for raw in PHRASE_SPLIT_RE.split(sentence):
        p = clean_phrase(raw)
        if not p or p in NOISE_PHRASES:
            continue
        if len(p) < 2 or len(p) > 30:
            continue
        # A real Chinese-label AE term always contains at least one Chinese
        # character; drug codes/regimens/study names ("Q4W", "AD-1225",
        # "CHRONOS") can pass the signal-keyword filter via a nearby comma
        # but are never themselves AE terms.
        if not HAS_CHINESE_RE.search(p):
            continue
        # Cross-reference remnants ("见药物相互作用", left over after 【】
        # stripping) are pointers to other sections, not AE content.
        if p.startswith("见"):
            continue
        phrases.append(p)
    return phrases


def main():
    if len(sys.argv) != 3:
        print("Usage: python prefilter.py <sections.json> <candidates.json>")
        sys.exit(1)
    sections_path, out_path = sys.argv[1], sys.argv[2]

    with open(sections_path, encoding="utf-8") as f:
        sections = json.load(f)

    candidates = []
    for sec in sections:
        for sentence in SENTENCE_SPLIT_RE.split(sec["text"]):
            sentence = sentence.strip()
            if not sentence or not looks_like_ae(sentence):
                continue
            for phrase in split_candidates(sentence):
                candidates.append({
                    "section": sec["section"],
                    "start_page": sec["start_page"],
                    "end_page": sec["end_page"],
                    "context_sentence": sentence,
                    "candidate_phrase": phrase,
                })

    # de-dup identical (section, phrase) pairs, keep first context
    seen = set()
    deduped = []
    for c in candidates:
        key = (c["section"], c["candidate_phrase"])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(c)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(deduped, f, ensure_ascii=False, indent=2)

    print(f"{len(candidates)} raw candidates -> {len(deduped)} deduped -> {out_path}")
    by_section = {}
    for c in deduped:
        by_section.setdefault(c["section"], 0)
        by_section[c["section"]] += 1
    for sec, n in sorted(by_section.items(), key=lambda x: -x[1]):
        print(f"  {sec}: {n}")


if __name__ == "__main__":
    main()
