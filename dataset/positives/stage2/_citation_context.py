#!/usr/bin/env python3
"""Extract citation sentences (contexts) for verified references from the base paper PDF, regex only.

Author-year styles: a sentence cites entry E if it contains E's first-author surname followed
(within a few tokens) by E's year, e.g. "(Huber, 1964)", "Huber (1964)", "Hirst et al., 2018",
"Assmann and Czaplicka, 1995".
Numeric styles: a sentence cites entry [n] if it contains a bracket group including n, e.g.
"[3]", "[1, 4]", "[2-5]".
Compares coverage with the contexts Semantic Scholar returned.
"""
import re
import unicodedata
import _reference_verification as vr

FRONT_MATTER = re.compile(r'Proceedings of the \d+(?:st|nd|rd|th) [^.]{0,120}?Association for Computational Linguistics[^.]{0,200}?(?:c⃝|©)\s*\d{4} Association for Computational Linguistics|c⃝\s*\d{4} Association for Computational Linguistics')


# running headers/footers of ML venues and arXiv that PyMuPDF merges into body sentences.
ML_FRONT_MATTER = re.compile('|'.join([
    r'(?:\b\d{1,3}\s+)?(?:Published|Under review) as a (?:conference|workshop) paper at [A-Z][A-Za-z]{2,10} \d{4}',
    r'Proceedings of the \d+(?:st|nd|rd|th) International Conference on [^.]{0,120}\.\s*PMLR:? (?:Volume )?\d+,\s*\d{4}\.(?:\s*Copyright \d{4} by the author\(s\)\.)?',
    r'PMLR \d+,\s*\d{4}\.',
    r'Copyright \d{4} by the author\(s\)\.',
    r'\d+(?:st|nd|rd|th) Conference on Neural Information Processing Systems \((?:NeurIPS|NIPS) \d{4}\)(?:[^.]{0,60})?\.',
    r'Preprint\.\s*Under review\.',
    r'arXiv:\d{4}\.\d{4,5}v\d+\s*\[[a-zA-Z\-.]+\]\s*\d{1,2}\s+[A-Z][a-z]{2}\s+\d{4}',
]))


def strip_front_matter(text):
    """Remove page-footer/header blocks that PyMuPDF merges into body text: the ACL proceedings footer ("Proceedings of the 57th
    Annual Meeting ... c 2019 ACL") and ICLR/ICML/NeurIPS headers/footers and the arXiv side stamp."""
    return re.sub(r'\s+', ' ', ML_FRONT_MATTER.sub(' ', FRONT_MATTER.sub(' ', text))).strip()


ABBR = re.compile(r'\b(et al|e\.g|i\.e|cf|vs|Fig|Eq|resp|approx|No|Vol|pp|Dr|Mr|Ms|Prof|Inc|Ltd|Jr|Sr|St)\.$')


def body_sentences(pdf_path):
    blocks = vr.column_blocks(pdf_path)
    heads = [i for i, t in enumerate(blocks) if re.match(r'\s*(References|REFERENCES|Bibliography)\b', t)]
    body = ' '.join(blocks[:heads[-1]] if heads else blocks)
    body = strip_front_matter(body)
    # Split on sentence end followed by space and an uppercase/quote/bracket start, protecting abbreviations.
    out, buf = [], ''
    body = re.sub(r'(?<=[.!?])\s+(\d{1,2}(?:\.\d{1,2}){0,2})\s+(?=[A-Z])', r' \1 ', body)  # keep heading numbers, split below
    for piece in re.split(r'(?<=[.!?])\s+(?=[A-Z"“(\[])|(?<=[.!?])\s+(?=\d{1,2}(?:\.\d{1,2}){0,2}\s+[A-Z])', body):
        buf = (buf + ' ' + piece).strip() if buf else piece
        if ABBR.search(buf):
            continue
        out.append(buf); buf = ''
    if buf:
        out.append(buf)
    return [s for s in out if 30 <= len(s) <= 1200]


CONTEXT_VERSION = 'v2-paragraph-windows'   # citing paragraph + previous/next paragraph; sentence-level contexts kept


def body_paragraphs(pdf_path):
    """Paragraph-like text blocks (PyMuPDF layout blocks in reading order) up to the References heading.
    A block that does not end a sentence is merged with the next one (column/page continuation)."""
    blocks = vr.column_blocks(pdf_path)
    heads = [i for i, t in enumerate(blocks) if re.match(r'\s*(References|REFERENCES|Bibliography)\b', t)]
    blocks = blocks[:heads[-1]] if heads else blocks
    out, buf = [], ''
    for b in blocks:
        b = strip_front_matter(b)
        if not b: continue
        buf = (buf + ' ' + b).strip() if buf else b
        if re.search(r'[.!?]["”)\]]?$', buf) or len(buf) > 3000:
            out.append(buf); buf = ''
    if buf: out.append(buf)
    return [p for p in out if len(p) >= 40]


def paragraph_windows(paras, hit, max_windows=3):
    """For paragraphs where hit(p) is true: {'before','citing','after'} with neighbouring paragraphs (None at the edges)."""
    idx = [i for i, p in enumerate(paras) if hit(p)][:max_windows]
    return [{'before': paras[i - 1] if i > 0 else None, 'citing': paras[i], 'after': paras[i + 1] if i + 1 < len(paras) else None} for i in idx]


SPACING_ACCENTS = re.compile(r'[´˜¨`ˆ˘˚¸˝˙ˇ]')


def fold(text):
    """Accent-insensitive form used only for matching: NFKD, combining marks and PDF spacing accents ("H´ebert") removed."""
    text = unicodedata.normalize('NFKD', SPACING_ACCENTS.sub('', text or ''))   # strip spacing accents first: NFKD turns '´' into ' ' + U+0301
    return ''.join(c for c in text if not unicodedata.combining(c))


NAME_SUFFIX = re.compile(r'(?:Jr|Sr|II|III|IV)\.?$')
PARTICLES = {'van', 'von', 'der', 'den', 'de', 'del', 'della', 'di', 'da', 'dos', 'du', 'la', 'le', 'ten', 'ter', 'zu'}


def first_surname(entry_raw, style=None):
    """First author's surname of a bibliography entry.
    Handles "Firstname Surname" (ACL, NeurIPS/ICLR natbib), "Surname, I." (ICML/AAAI), "I. Surname", "Surname IJ" (Vancouver),
    organisation authors ("OpenAI. GPT-4 technical report"), leading "[n]" / "n." labels whatever the detected style,
    and non-ASCII surnames (letters are kept; matching is accent-folded)."""
    e = re.sub(r'^\s*(?:\[\d{1,3}\]|\d{1,3}\.)\s*', '', entry_raw.strip())
    e = re.sub(r'^\[[^\]]{2,80}?(?:19|20)\d{2}[a-z]?\]\s*', '', e)   # IJCAI "[Raychev et al., 2015] Veselin Raychev, ..."
    head = re.split(r'\.\s(?:19|20)\d{2}[a-z]?\.|:\s|\(\s*(?:19|20)\d{2}', e, 1)[0]
    first = re.split(r';|\s+and\s+|\s*&\s*|,', head, 1)[0].strip()
    cut = re.match(r'(.+?[^\W\d_]{2,})\.\s', first)   # author block ends at a period after a 2+ letter word, not after an initial
    if cut:
        first = cut.group(1)
    toks = [t for t in re.split(r'\s+', first) if t and not NAME_SUFFIX.fullmatch(t)]
    # drop lowercase surname particles so "van der Aalst W" gives Aalst, matching "van der Aalst (2011)" in the text
    toks = [t for t in toks if t.lower() not in PARTICLES or t[:1].isupper()] or toks
    if not toks:
        return None
    # "Surname IJ" / "Surname I." -> first token; otherwise the last token ("Xinran Gu", "X. Gu", "Martínez Beltrán").
    cand = toks[0] if len(toks) > 1 and re.fullmatch(r'(?:[A-Z]\.?-?){1,3}', toks[-1]) else toks[-1]
    cand = re.sub(r"[^\w\-'’]|[\d_]", '', cand)
    return cand if len(cand) >= 2 else None


NAME = r"[A-Z][\w'’\-\.]*"
AUTHOR = r"(?:%s\s+){1,4}[A-Z][\w'’\-]+" % NAME          # "Eduard H. Hovy", "Jimmy Lei Ba"
AUTHORS_END = re.compile(r"(%s(?:,\s+%s)*(?:,?\s+and\s+%s)?)\s*$" % (AUTHOR, AUTHOR, AUTHOR))


def author_block_before(text):
    """Trailing author list of `text` (which should end right before '. YEAR.'), or None."""
    mm = AUTHORS_END.search(text.strip())
    return mm.group(1) if mm else None


# A year token of a bibliography entry: not part of an arXiv id ("2012.01234"), a DOI ("j.physa.2006.03.050") or a longer number.
ENTRY_YEAR = re.compile(r'(?<![\d./:])((?:19|20)\d{2})([a-z]?)(?![\d]|\.\d)')


def entry_years(entry_raw):
    """Candidate citation years of an entry, with letter suffixes ("2024b"), in order of appearance."""
    return tuple(dict.fromkeys(y + s for y, s in ENTRY_YEAR.findall(entry_raw)))


def locator_for_title(entry_raw, title, style):
    """(surname, years) of the author block immediately before `title` inside a (possibly merged) entry.

    `years` is a tuple of year strings with letter suffixes.
      * ACL form "Authors. 2019a. Title." -> that single year (suffix now kept, so 2019a and 2019b are told apart);
      * otherwise (year at the end, NeurIPS/ICLR/ICML: "Authors. Title. In ICLR, 2023.") -> every year token of the entry,
        so the in-text year equals one of the entry's years.
    The surname uses the ACL author block when present, else first_surname()."""
    if style in ('numeric', 'lncs'):
        return None
    flat = re.sub(r'\s+', ' ', entry_raw)
    key = re.sub(r'\s+', ' ', (title or '')[:40]).lower()
    i = flat.lower().find(key)
    head = flat[:i] if i > 0 else flat
    mk = re.search(r'\.\s((?:19|20)\d{2})([a-z]?)\.\s*$', head)
    if not mk:
        marks = list(re.finditer(r'\.\s((?:19|20)\d{2})([a-z]?)\.\s', head))
        if not marks:
            return (first_surname(entry_raw, style), entry_years(entry_raw))
        mk = marks[-1]
    year = (mk.group(1) + mk.group(2),)
    authors = author_block_before(fold(head[:mk.start()]))   # PDF spacing accents ("Petrovi´c") broke the author regex
    if not authors:
        return (first_surname(entry_raw, style), year)
    first = re.split(r',| and ', authors, 1)[0].strip()
    toks = [t for t in first.split() if t]
    surname = re.sub(r"[^\w\-'’]|[\d_]", '', toks[-1]) if toks else None
    return (surname if surname and len(surname) >= 2 else None, year)


AY_VERSION = 3
# same-author year lists, "Liu et al., 2016, 2017" / "Liu et al. (2016a, 2017)" / "Liu et al., 2016a,b, 2017".
# "&" between authors ("Kolluru & Kitchin, 2024"), non-ASCII co-author names, possessive ("Chai et al.'s (2024)"),
#   "[" before the year, ";" in same-author year lists, several candidate years, exact letter suffix when the entry has one,
#   accent-folded matching (use cites()).
CO_NAME = r"[^\W\d_][\w'’\-]*"


def _year_alt(year):
    y = str(year)
    if len(y) == 5:   # "2024b": also inside "2024a,b"
        return y[:4] + r'(?:[a-z]\s*[,;]\s*)*' + y[4] + r'\b'   # "2022b;a" (natbib compressed lists) too
    return y + r'[a-z]?\b'


def author_year_pattern(surname, year):
    """Surname (+ et al. / and|& X / X, Y and|& Z, possessive), optional other years of the same group, then one of `year`.
    `year` is a year (int or "2019"/"2019a") or a tuple of them. Search with cites(), which accent-folds the sentence."""
    years = year if isinstance(year, (tuple, list)) else (year,)
    s = re.escape(fold(surname))
    co = r"(?:\s+et\s+al\.?|\s+(?:and|&)\s+%s(?:\s+%s)?|,\s+%s,?\s+(?:and|&)\s+%s)?" % (CO_NAME, CO_NAME, CO_NAME, CO_NAME)
    prior = r"(?:(?:19|20)\d{2}[a-z]?(?:\s*,\s*[a-z]\b)*\s*[,;]\s*)*"
    alt = '(?:' + '|'.join(_year_alt(y) for y in years) + ')'
    return re.compile(r'(?<!\w)' + s + co + r"(?:['’]s)?[,\s]*[\(\[]?\s*" + prior + alt, re.I)


def cites(pattern, text):
    """Does `text` contain the author-year citation `pattern`? (accent-folded, so "Hébert", "H´ebert" and "Hebert" agree)"""
    return bool(pattern.search(fold(text)))


# a superscript number must follow a lowercase letter or closing punctuation, never an uppercase letter ("S12",
# "H2") or a decimal point ("Theorem 2.8", "eta = 0.3"); see also citation_mode().
SUPERSCRIPT = re.compile(r'(?<=[a-z\)\]\.,;:])(?<!\d\.)(?<!\d,)(\d{1,3}(?:\s*[,–-]\s*\d{1,3})*)(?=[\s\.,;:\)]|$)')


def citation_mode(sentences):
    """'bracket' when the body cites with [n] at least 5 times (then glued numbers are never citations: IEEE/Elsevier/arXiv
    papers have "S12 = S15", "n = 400"), else 'superscript'."""
    return 'bracket' if sum(1 for x in sentences if re.search(r'\[\s*\d{1,3}\s*(?:[,;\]–\-])', x)) >= 5 else 'superscript'


def numeric_ids(sentence, max_id=None, allow_superscript=True):
    """Citation numbers in a sentence: bracketed "[3]", "[1, 4]", "[2-5]" always; when the sentence has no brackets and
    max_id is given, also superscript-style numbers glued to the preceding word ("...previously.12", "...shown,4,15")
    that are <= max_id (biomedical / medical styles)."""
    ids = set()
    for grp in re.findall(r'\[([\d,;\s\-–]+)\]', sentence):   # also ";" separators ("[60; 86]")
        for part in re.split(r'[,;\s]+', grp):
            if not part:
                continue
            rng = re.fullmatch(r'(\d+)[\-–](\d+)', part)
            if rng:
                ids.update(range(int(rng.group(1)), int(rng.group(2)) + 1))
            elif part.isdigit():
                ids.add(int(part))
    if not ids and max_id and allow_superscript:
        for grp in SUPERSCRIPT.findall(sentence):
            for part in re.split(r'\s*,\s*', grp):
                rng = re.fullmatch(r'(\d+)\s*[\-–]\s*(\d+)', part)
                vals = range(int(rng.group(1)), int(rng.group(2)) + 1) if rng else ([int(part)] if part.isdigit() else [])
                ids.update(v for v in vals if 1 <= v <= max_id)
    return ids
