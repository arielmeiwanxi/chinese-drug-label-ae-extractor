"""Stage 1: split the LPI PDF into section-tagged text.

PDF text extraction breaks text at the page's right margin (word-wrap), not
at real sentence/paragraph/table-row boundaries. If we naively treat every
line break as meaningful, mid-word wraps ("速发过敏\n反应") get corrupted and
real list/table-row boundaries get lost. We use each line's right-edge
x-coordinate to tell the two apart: a line whose text reaches close to the
page's content-width is a wrapped continuation (glue to the next line with
no separator); a line that stops well short of that margin is a genuine
line end (paragraph/table-row/heading) and becomes a real line break.

Usage: python extract_sections.py <input.pdf> <output.json>
"""
import json
import re
import sys

import pdfplumber

FOOTER_RE = re.compile(r"^\d+\s*/\s*\d+$")
# A real section heading occupies its own (reflowed) line, e.g. "【不良反应】"
# with nothing else on that line. Inline cross-references like
# "（见【不良反应】）" or enumerations like "【用法用量】、【不良反应】和【临床
# 药理】。" share a line with other text and must NOT be treated as boundaries.
HEADING_LINE_RE = re.compile(r"^【([^】]+)】$")

WRAP_MARGIN = 25  # points below the document's max line width still counted as "full width"


def extract_reflowed_lines(pdf_path):
    """Return a list of (page_num, text) for each *genuine* line, with
    page-width word-wraps already glued back together."""
    raw_lines = []  # (page_num, x1, text)
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            for l in page.extract_text_lines():
                text = l["text"].strip()
                if not text or FOOTER_RE.match(text):
                    continue
                raw_lines.append((i + 1, l["x1"], text))

    if not raw_lines:
        return []

    max_x1 = max(x1 for _, x1, _ in raw_lines)
    wrap_threshold = max_x1 - WRAP_MARGIN

    reflowed = []
    buf, buf_page = "", None
    for page_num, x1, text in raw_lines:
        if buf_page is None:
            buf_page = page_num
        buf += text
        if x1 < wrap_threshold:
            reflowed.append((buf_page, buf))
            buf, buf_page = "", None
    if buf:
        reflowed.append((buf_page, buf))
    return reflowed


def split_into_sections(reflowed_lines):
    """Walk the reflowed lines, bucketing text under the most recent heading
    line seen. Only a line that consists SOLELY of 【标题】 counts as a section
    boundary."""
    sections = []
    current_name = "(document start, no heading)"
    current_start_page = reflowed_lines[0][0] if reflowed_lines else 1
    current_text = []

    def flush(end_page):
        joined = "\n".join(current_text).strip()
        if joined:
            sections.append({
                "section": current_name,
                "start_page": current_start_page,
                "end_page": end_page,
                "text": joined,
            })

    last_page = current_start_page
    for page_num, text in reflowed_lines:
        last_page = page_num
        m = HEADING_LINE_RE.match(text)
        if m:
            flush(page_num)
            current_name = m.group(1)
            current_start_page = page_num
            current_text = []
        else:
            current_text.append(text)

    flush(last_page)
    return sections


def main():
    if len(sys.argv) != 3:
        print("Usage: python extract_sections.py <input.pdf> <output.json>")
        sys.exit(1)
    pdf_path, out_path = sys.argv[1], sys.argv[2]

    reflowed = extract_reflowed_lines(pdf_path)
    sections = split_into_sections(reflowed)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(sections, f, ensure_ascii=False, indent=2)

    print(f"Extracted {len(sections)} sections from {len(reflowed)} reflowed lines -> {out_path}")
    for s in sections:
        print(f"  [{s['start_page']}-{s['end_page']}] {s['section']} ({len(s['text'])} chars)")


if __name__ == "__main__":
    main()
