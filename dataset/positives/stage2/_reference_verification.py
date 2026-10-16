#!/usr/bin/env python3
"""Verify Semantic Scholar reference links against the base paper's own bibliography (no LLM).

1. Fetch the base paper's open PDF (reuses fulltext.FullTextClient; arXiv/ACL Anthology PDFs).
2. Locate the References section and split it into entries with regexes
   (author-year styles such as ACL/AAAI, and numeric "[12]" styles).
3. Extract each entry's year and title.
4. For every target returned by Semantic Scholar /references, compute the S2ORC title
   similarity (harmonic mean of Jaccard and containment over character 3-grams of the
   normalized title) against all parsed entries. A link is VERIFIED if the best score
   is >= 0.8 (S2ORC's threshold).
Outputs per-base parse stats and per-target verification records.
"""
import re

YEAR = r'(?:19|20)\d{2}[a-z]?'


def norm_title(t):
    return re.sub(r'[^a-z0-9]', '', (t or '').lower())


def grams(s, n=3):
    return {s[i:i + n] for i in range(len(s) - n + 1)} if len(s) >= n else {s}


def containment(title, entry_text):
    """Share of the title's character 3-grams found in the entry text; guards against very short titles."""
    A = grams(norm_title(title))
    if len(norm_title(title)) < 20 or not A:
        return 0.0
    return len(A & grams(norm_title(entry_text))) / len(A)


def s2orc_sim(a, b):
    A, B = grams(norm_title(a)), grams(norm_title(b))
    if not A or not B:
        return 0.0
    inter = len(A & B)
    j = inter / len(A | B)
    c = inter / min(len(A), len(B))
    return 0.0 if j + c == 0 else 2 * j * c / (j + c)


def column_blocks(pdf_path):
    """Text blocks in reading order for 1- or 2-column pages (used for body text)."""
    import pymupdf
    blocks = []
    with pymupdf.open(pdf_path) as doc:
        for pno, page in enumerate(doc):
            mid = page.rect.width / 2
            raw = [b for b in page.get_text('blocks') if b[6] == 0 and b[4].strip()]
            two_col = sum((b[2] - b[0]) < mid * 1.1 for b in raw) > 0.6 * max(len(raw), 1)
            for b in raw:
                col = (0 if not two_col else (0 if b[0] < mid - 20 else 1))
                blocks.append((pno, col, b[1], re.sub(r'-\n(?=[a-z])', '', b[4]).replace('\n', ' ').strip()))
    blocks.sort(key=lambda x: (x[0], x[1], x[2]))
    return [t for *_, t in blocks]


def column_lines(pdf_path):
    """Individual text lines with x0, in reading order: (page, col, y0, x0, text)."""
    import pymupdf
    lines = []
    with pymupdf.open(pdf_path) as doc:
        for pno, page in enumerate(doc):
            mid = page.rect.width / 2
            d = page.get_text('dict')
            blocks = [b for b in d['blocks'] if b.get('type') == 0]
            two_col = sum((b['bbox'][2] - b['bbox'][0]) < mid * 1.1 for b in blocks) > 0.6 * max(len(blocks), 1)
            for b in blocks:
                for ln in b['lines']:
                    text = ''.join(sp['text'] for sp in ln['spans']).strip()
                    if not text:
                        continue
                    x0, y0 = ln['bbox'][0], ln['bbox'][1]
                    col = 0 if not two_col else (0 if x0 < mid - 20 else 1)
                    lines.append((pno, col, y0, x0, text))
    lines.sort(key=lambda x: (x[0], x[1], x[2]))
    return lines


NAME_RE = r"[A-Z][\w'’\-\.]*"
AUTHOR_RE = r"(?:%s\s+){1,4}[A-Z][\w'’\-]+" % NAME_RE
# A record boundary: preceded by a sentence end that is NOT an initial ("M. E. J. Newman"), followed by an author
# list whose first token is a full first name (2+ letters), then ". YEAR. ".
FIRSTNAME_RE = r"[A-Z][a-z][\w'’\-]*"
FIRST_AUTHOR_RE = r"%s(?:\s+%s){0,3}\s+[A-Z][\w'’\-]+" % (FIRSTNAME_RE, NAME_RE)
AUTHORS_YEAR_RE = re.compile(r"(?<=[.)\]] )(?<![A-Z]\. )(%s(?:,\s+%s)*(?:,?\s+and\s+%s)?\.\s(?:19|20)\d{2}[a-z]?\.\s)" % (FIRST_AUTHOR_RE, AUTHOR_RE, AUTHOR_RE))


def split_merged_author_year(entry):
    """If several 'Authors. YEAR. Title...' records were merged into one entry, split them."""
    starts = [mm.start(1) for mm in AUTHORS_YEAR_RE.finditer(entry)]
    starts = [s for s in starts if s > 20]
    if not starts:
        return [entry]
    cuts = [0] + starts + [len(entry)]
    return [entry[a:b].strip() for a, b in zip(cuts, cuts[1:]) if entry[a:b].strip()]


def reference_entries(pdf_path):
    """Segment the References section into entries using hanging indents / numeric labels.

    An entry starts on a line whose x0 sits at the column's left margin (continuation
    lines are indented), or on a line beginning with "[n]" / "n." labels.
    """
    lines = column_lines(pdf_path)
    HEAD = r'(References|REFERENCES|Reference List|Bibliography|BIBLIOGRAPHY|Literature Cited|LITERATURE CITED|Works Cited|References and Notes|REFERENCES AND NOTES|Notes and References|Referencias|REFERENCIAS|Referências|REFERÊNCIAS|Bibliografía|BIBLIOGRAFÍA|Bibliografia|BIBLIOGRAFIA|Literatur|LITERATUR|Literaturverzeichnis|Bibliographie|BIBLIOGRAPHIE|Références|RÉFÉRENCES|Referensi|REFERENSI|Daftar Pustaka|DAFTAR PUSTAKA|Literatura Citada|LITERATURA CITADA|Riferimenti bibliografici)'
    heads = [i for i, l in enumerate(lines) if re.fullmatch(r'\s*(?:\d{1,2}\.?\s+)?' + HEAD + r'\s*[:.]?\s*', l[4])]  # multilingual headings, optional numbering/colon
    if not heads:
        return None, None
    region = []
    for l in lines[heads[-1] + 1:]:
        t = l[4]
        if re.match(r'\s*(Appendix|APPENDIX|Appendices|Supplementary Material|Supplemental Material|Ethics Statement|Ethical Considerations|Limitations)\b', t) and len(t) < 60:
            break
        if re.fullmatch(r'\s*[A-Z]\s{1,3}[A-Z][A-Za-z ,:\-]{3,60}', t):  # "A  Additional Results" style appendix heading
            break
        # "A Proof of Theorem 1" / "A.1 Lemmas" appendix headings (single space, no year) after the list
        if region and len(t) < 70 and not re.search(YEAR, t) and re.fullmatch(r'\s*(?:[A-Z]|[A-Z]\.\d{1,2})\s+[A-Z][a-z]{2,}[\w ,:\-]*', t):
            break
        if re.fullmatch(r'\s*\d{1,5}\s*', t):
            continue  # page number
        region.append(l)
    if not region:
        return [], 'unknown'
    # Left margin per (page, column): the smallest x0 that at least two lines share (within 1.5pt).
    margins = {}
    for key in {(l[0], l[1]) for l in region}:
        xs = sorted(l[3] for l in region if (l[0], l[1]) == key)
        margins[key] = min(xs)
    entries, cur = [], ''
    margin_starts = sum((l[3] - margins[(l[0], l[1])]) < 3.0 for l in region)
    numeric_labels = sum(bool(re.match(r'\[\d{1,3}\]', l[4])) for l in region)
    lncs_labels = sum(bool(re.match(r'\d{1,3}\.\s+[A-Z]', l[4])) for l in region)  # any capitalised start ("1. Hayes JD, ..." biomedical lists), not only "Surname,"
    # labels that run 1, 2, 3, ... define the entries even when margin statistics are off (appendix pages after the
    # list, a stray line left of the labels); checked before the margin test below.
    def _sequential(rx):
        nums = [int(m.group(1)) for m in (re.match(rx, l[4]) for l in region) if m]
        return len(nums) >= 5 and sum(b == a + 1 for a, b in zip(nums, nums[1:])) >= 0.8 * (len(nums) - 1)
    seq_numeric, seq_lncs = _sequential(r'\[(\d{1,3})\]'), _sequential(r'(\d{1,3})\.\s+[A-Z]')
    # Labels define entries only when they account for most margin-aligned lines.
    if numeric_labels < 0.5 * max(margin_starts, 1) and not seq_numeric:
        numeric_labels = 0
    if lncs_labels < 0.5 * max(margin_starts, 1) and not seq_lncs:
        lncs_labels = 0
    for l in region:
        t = l[4]
        is_start = (l[3] - margins[(l[0], l[1])]) < 3.0
        if numeric_labels >= 5:
            is_start = bool(re.match(r'\[\d{1,3}\]', t))
        elif lncs_labels >= 5:
            is_start = bool(re.match(r'\d{1,3}\.\s+[A-Z]', t))
        if is_start and cur:
            entries.append(cur); cur = t
        else:
            cur = (cur[:-1] + t) if cur.endswith('-') and t[:1].islower() else (cur + ' ' + t if cur else t)
    if cur:
        entries.append(cur)
    entries = [re.sub(r'\s+', ' ', e).strip() for e in entries if re.search(YEAR, e)]
    style = 'numeric' if numeric_labels >= 5 else 'lncs' if lncs_labels >= 5 else 'author-year'
    # NeurIPS-style "[n]" labels whose continuation lines align with the label start every margin line, so the
    # line-count test above misses them; decide on the segmented entries instead.
    if style == 'author-year' and len(entries) >= 5 and sum(bool(re.match(r'\[\d{1,3}\]', e)) for e in entries) >= 0.6 * len(entries):
        style = 'numeric'
    # MDPI-style "1." labels sit on their own line, so the line test above misses them; decide on the entries.
    if style == 'author-year' and len(entries) >= 5 and sum(bool(re.match(r'\d{1,3}\.\s+\S', e)) for e in entries) >= 0.6 * len(entries):
        style = 'lncs'
    if style == 'author-year':
        entries = [part for e in entries for part in split_merged_author_year(e)]
    if style == 'author-year' and len(re.findall(r', ' + YEAR + r'\.$', ' '.join(e[-12:] for e in entries))) > 0.5 * len(entries):
        style = 'year-end'
    return entries, style


QUOTED_TITLE = re.compile(r'(?:‘‘|“|``|"|«)\s*(.{8,400}?)\s*[,.]?\s*(?:’’|”|\'\'|"|»)')


def parse_entry(e):
    """a title in typographic quotes (IEEE ‘‘Title,’’, APA-like “Title”) is taken as the title first."""
    r = _parse_entry(e)
    qm = QUOTED_TITLE.search(e)
    if r is not None and qm:
        r['title'] = qm.group(1).strip().rstrip(',.')
    return r


def _parse_entry(e):
    mm = re.search(r'\.\s(' + YEAR + r')\.\s(.+)', e)
    lm = re.match(r'\d{1,3}\. (.+?): (.+)', e)  # LNCS: "n. Authors: Title. In: ..."
    if lm and not mm:
        ym = re.search(r'\((' + YEAR + r')\)|\b(' + YEAR + r')\b', e)
        year = (ym.group(1) or ym.group(2)) if ym else '0000'
        rest = lm.group(2)
    elif mm:
        year, rest = mm.group(1), mm.group(2)
    else:
        mm = re.search(r'\((' + YEAR + r')\)\.?\s(.+)', e)
        if mm:
            year, rest = mm.group(1), mm.group(2)
        else:
            ym = re.search(r'\b(' + YEAR + r')\b', e)
            if not ym:
                return None
            year = ym.group(1)
            # Year-at-end style: title follows the author block ("... and S. Sanghai. Title. Venue, 2023.")
            am = re.search(r'(?:[A-Z]\.|[A-Z][a-z]+)\.\s+(?=[A-Z])', e)
            rest = e[am.end():] if am else e
    # Title = up to the first period followed by a space and a venue-ish token.
    tm = re.match(r'(.+?[.?!])\s(?=(?:In |Proceedings|Proc\.|arXiv|Journal|Trans\.|IEEE|ACM|[A-Z][a-z]+ (?:of|for|on) |[A-Z][A-Za-z]+(?:,|\s\d)|\d))', rest)
    title = (tm.group(1) if tm else rest.split('. ')[0]).strip().rstrip('.')
    return {'year': int(year[:4]) if year[:4].isdigit() else 0, 'title': title, 'raw': e[:300], 'raw_full': e}


def in_text_match(contexts, target):
    """Secondary check: do any (Surname, Year) markers in the citing sentences match the target record?"""
    cites = re.findall(r'\(([^()]*?' + YEAR + r')\)', ' '.join(contexts))
    if not cites:
        return None
    surnames = {a['name'].split()[-1].lower() for a in (target.get('authors') or []) if a.get('name')}
    y = target.get('year') or 0
    for grp in cites:
        for c in grp.split(';'):
            sm = re.match(r'\s*([A-Z][A-Za-z\-\'’]+)', c.strip()); ym = re.search(r'(\d{4})', c)
            if sm and ym and sm.group(1).lower() in surnames and abs(int(ym.group(1)) - y) <= 2:
                return True
    return False


def author_year_match(entry, target, year_tol=2):
    """Metadata check on top of the title match: the Semantic Scholar record's author surname(s) must appear in the
    reference entry text, and the entry year must be within `year_tol` of the record year. Catches records whose
    title matches but that point to a different work (e.g. two textbooks both titled "Convex Optimization").
    Returns (author_ok, year_ok); author_ok is None when the record lists no authors."""
    if entry is None:
        return None, None
    text = (entry.get('raw_full') or entry.get('raw') or '').lower()
    surnames = [a['name'].split()[-1].lower() for a in (target.get('authors') or []) if a.get('name') and len(a['name'].split()[-1]) > 1]
    author_ok = None if not surnames else any(re.search(r'\b' + re.escape(sn) + r'\b', text) for sn in surnames)
    ey, ty = entry.get('year') or 0, target.get('year') or 0
    year_ok = None if not (ey and ty) else abs(ey - ty) <= year_tol
    return author_ok, year_ok


def link_status(title_ok, author_ok, year_ok):
    """verified = title matches and neither author nor year contradicts the record.
    metadata_mismatch = title matches but author or year contradicts (record likely points at a different work)."""
    if not title_ok:
        return 'unverified'
    if author_ok is False or year_ok is False:
        return 'metadata_mismatch'
    return 'verified'
