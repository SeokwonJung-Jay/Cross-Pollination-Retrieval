"""Semantic Scholar s2orc_v2 full text (Grobid-parsed, snapshot 2026-09-22): reference list + citing sentences from its annotations.

Every selected base paper has an s2orc_v2 body (selection rule 9). body.annotations.bib_ref spans carry ref_id (-> a
bibliography.annotations.bib_entry id) and sometimes matched_paper_id (Semantic Scholar corpus id). Sentences come from the
'sentence' annotation when present, otherwise each paragraph is split with the same rule as the JATS path. Ranges "[3-5]" whose two
end markers are separated only by a dash add the entries between them (bibliography order).
"""
from __future__ import annotations

import json
import re

ABBR = re.compile(r'\b(et al|e\.g|i\.e|cf|vs|Fig|Figs|Eq|Eqs|Ref|Refs|resp|approx|ca|No|Vol|pp|Dr|Mr|Ms|Prof|Inc|Ltd|Jr|Sr|St|sp|spp|var|subsp|al)\.$')
DASH = re.compile(r'^\s*[–—\-−]\s*$')
YEAR = re.compile(r'\b(19|20)\d{2}\b')


class S2orcUnavailable(Exception):
    pass


def _ann(block, key):
    v = (block.get('annotations') or {}).get(key)
    if not v:
        return []
    out = json.loads(v) if isinstance(v, str) else v
    for a in out:   # some records store offsets as strings
        a['start'], a['end'] = int(a['start']), int(a['end'])
    return out


def references(rec):
    bib = rec.get('bibliography') or {}
    text = bib.get('text') or ''
    titles = [(a['start'], a['end']) for a in _ann(bib, 'bib_title')]
    out = []
    for e in _ann(bib, 'bib_entry'):
        s, t = e['start'], e['end']; raw = re.sub(r'\s+', ' ', text[s:t]).strip()
        tt = next((text[a:b] for a, b in titles if a >= s and b <= t), '')
        ym = YEAR.search(raw)
        attrs = e.get('attributes') or {}
        out.append({'id': attrs.get('id'), 'raw_full': raw, 'title': re.sub(r'\s+', ' ', tt).strip().rstrip('.'),
                    'year': int(ym.group(0)) if ym else 0, 'matched_paper_id': attrs.get('matched_paper_id')})
    return out


def _split(text, base):
    """[(start, end)] sentence spans of text (offsets + base)."""
    spans, start = [], 0
    for m in re.finditer(r'(?<=[.!?])\s+(?=[A-Z"“(\[])', text):
        if ABBR.search(text[start:m.start()]):
            continue
        spans.append((base + start, base + m.start())); start = m.end()
    spans.append((base + start, base + len(text)))
    return spans


def citing_sentences(rec, refs):
    body = rec.get('body') or {}
    text = body.get('text') or ''
    order = {r['id']: i for i, r in enumerate(refs)}
    brefs = sorted(((a['start'], a['end'], (a.get('attributes') or {}).get('ref_id'), (a.get('attributes') or {}).get('matched_paper_id'))
                    for a in _ann(body, 'bib_ref')), key=lambda x: x[0])
    for b in brefs:   # carry a marker-level matched_paper_id onto its entry when the entry has none
        if b[2] in order and b[3] and not refs[order[b[2]]].get('matched_paper_id'):
            refs[order[b[2]]]['matched_paper_id'] = b[3]
    sents = [(a['start'], a['end']) for a in _ann(body, 'sentence')]
    if not sents:
        for p in _ann(body, 'paragraph'):
            sents += _split(text[p['start']:p['end']], p['start'])
    out = {}
    j = 0
    for s, t in sorted(sents):
        while j < len(brefs) and brefs[j][1] <= s:
            j += 1
        k, inside = j, []
        while k < len(brefs) and brefs[k][0] < t:
            if brefs[k][0] >= s: inside.append(brefs[k])
            k += 1
        if not inside:
            continue
        rids = {b[2] for b in inside if b[2]}
        for a, b in zip(inside, inside[1:]):
            if a[2] in order and b[2] in order and DASH.match(text[a[1]:b[0]]) and order[b[2]] > order[a[2]] + 1:
                rids.update(r['id'] for r in refs[order[a[2]] + 1:order[b[2]]])
        clean = re.sub(r'\s+', ' ', text[s:t]).strip()
        if 30 <= len(clean) <= 1500:
            for rid in rids:
                lst = out.setdefault(rid, [])
                if clean not in lst: lst.append(clean)
    return out


def parse(rec_text):
    rec = json.loads(rec_text) if isinstance(rec_text, str) else rec_text
    refs = references(rec)
    if not refs:
        raise S2orcUnavailable('s2orc_no_bibliography')
    sents = citing_sentences(rec, refs)
    if not sents:
        raise S2orcUnavailable('s2orc_no_body_citations')
    return refs, sents
