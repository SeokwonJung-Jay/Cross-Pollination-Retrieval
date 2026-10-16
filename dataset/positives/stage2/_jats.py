"""Europe PMC full-text JATS XML: reference list + in-text citation sentences, linked by the XML itself (no layout parsing).

PDFs of MDPI / Elsevier / Wiley / Hindawi are blocked (HTTP 403) and
PMC pages are HTML, while most open-access biomedical papers are in PubMed Central as JATS. In JATS every in-text citation is
<xref ref-type="bibr" rid="..."> pointing at a <ref id="..."> entry, so a citing sentence is assigned to its reference exactly.
Ranges written "[3–5]" carry only the two end xrefs; the refs between them (reference-list order) are added.
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

from lxml import etree

API = 'https://www.ebi.ac.uk/europepmc/webservices/rest/{pmcid}/fullTextXML'
MARK = re.compile(r'(\d+)')
DASH = re.compile(r'^\s*[–—\-−]\s*$')
ABBR = re.compile(r'\b(et al|e\.g|i\.e|cf|vs|Fig|Figs|Eq|Eqs|Ref|Refs|resp|approx|ca|No|Vol|pp|Dr|Mr|Ms|Prof|Inc|Ltd|Jr|Sr|St|sp|spp|var|subsp|al)\.$')
SKIP = {'table-wrap', 'fig', 'ref-list', 'disp-formula', 'supplementary-material', 'table', 'caption'}


class JatsUnavailable(Exception):
    pass


def fetch(pmcid, cache_dir, session=None, min_interval=0.2):
    """Return the JATS XML text for a PMCID ('PMC123' or '123'), cached on disk. Raises JatsUnavailable."""
    import requests
    pmcid = 'PMC' + str(pmcid).upper().removeprefix('PMC')
    cache_dir = Path(cache_dir); cache_dir.mkdir(parents=True, exist_ok=True)
    path, miss = cache_dir / f'{pmcid}.xml', cache_dir / f'{pmcid}.missing'
    if path.exists():
        return path.read_text()
    if miss.exists():
        raise JatsUnavailable(miss.read_text().strip() or 'jats_missing')
    s = session or requests
    for attempt in range(4):   # 429 / 5xx only: request not served, waiting costs nothing
        r = s.get(API.format(pmcid=pmcid), timeout=60)
        if r.status_code == 200 and '<article' in r.text[:2000]:
            path.write_text(r.text); time.sleep(min_interval); return r.text
        if r.status_code in (429, 502, 503, 504) and attempt < 3:   # Europe PMC answers 500 when no full text exists (not transient)
            time.sleep(10 * (attempt + 1)); continue
        reason = f'jats_http_{r.status_code}' if r.status_code != 200 else 'jats_not_xml'
        miss.write_text(reason); raise JatsUnavailable(reason)


def _text(el):
    return re.sub(r'\s+', ' ', ''.join(el.itertext())).strip() if el is not None else ''


def references(root):
    """[{id, label, title, year, surnames, doi, pmid, raw_full}] in reference-list order."""
    out = []
    for ref in root.iter('ref'):
        cit = next((c for c in ref if c.tag in ('element-citation', 'mixed-citation', 'citation', 'nlm-citation')), None)
        if cit is None:
            cit = ref
        title = _text(cit.find('.//article-title')) or _text(cit.find('.//chapter-title')) or _text(cit.find('.//source'))
        ym = re.search(r'(19|20)\d{2}', _text(cit.find('.//year')))
        ids = {p.get('pub-id-type'): _text(p) for p in cit.iter('pub-id')}
        doi = (ids.get('doi') or '').lower()
        if not doi:
            dm = re.search(r'10\.\d{4,9}/\S+', _text(cit)); doi = dm.group(0).rstrip('.').lower() if dm else ''
        out.append({'id': ref.get('id'), 'label': _text(ref.find('label')), 'title': title.rstrip('.'),
                    'year': int(ym.group(0)) if ym else 0, 'surnames': [_text(s) for s in cit.iter('surname')],
                    'doi': doi, 'pmid': ids.get('pmid') or '',
                    'raw_full': re.sub(r'\s+', ' ', ' '.join(cit.itertext())).strip()})   # spaces between tags: "Costa G.", not "CostaG."
    return out


def _paragraph_text(p, marks):
    """Text of a <p> with every bibr xref written as MARK + its label; table/figure content inside the paragraph is skipped.
    marks[i] = (rids, label)."""
    parts = []
    def walk(el):
        if el.tag in SKIP:
            if el.tail: parts.append(el.tail)
            return
        if el.tag == 'xref' and el.get('ref-type') == 'bibr':
            marks.append(((el.get('rid') or '').split(), _text(el))); parts.append(f'\uE000{len(marks) - 1}\uE001{_text(el)}')
            if el.tail: parts.append(el.tail)
            return
        if el.text: parts.append(el.text)
        for c in el: walk(c)
        if el is not p and el.tail: parts.append(el.tail)
    walk(p)
    return re.sub(r'\s+', ' ', ''.join(parts)).strip()


def _split(text):
    out, buf = [], ''
    for piece in re.split(r'(?<=[.!?])\s+(?=[A-Z"“(\[\uE000])', text):
        buf = (buf + ' ' + piece).strip() if buf else piece
        if ABBR.search(MARK.sub('', buf)):
            continue
        out.append(buf); buf = ''
    if buf: out.append(buf)
    return out


def citing_sentences(root, refs):
    """{ref_id: [sentence, ...]} for every reference cited in the body (markers removed, xref labels kept)."""
    order = {r['id']: i for i, r in enumerate(refs)}
    body = root.find('.//body')
    if body is None:
        return {}
    out = {}
    for p in body.iter('p'):
        if any(a.tag in SKIP for a in p.iterancestors()):
            continue
        marks = []; text = _paragraph_text(p, marks)
        if not marks:
            continue
        for sent in _split(text):
            found = [(int(m.group(1)), m.start(), m.end()) for m in MARK.finditer(sent)]
            rids = set()
            for k, (i, st, en) in enumerate(found):
                rids.update(marks[i][0])
                if k + 1 < len(found):   # "[3–5]": xref 3, dash, xref 5 -> add the refs between them in list order
                    j, st2 = found[k + 1][0], found[k + 1][1]
                    between = sent[en + len(marks[i][1]):st2]
                    a, b = marks[i][0][-1:], marks[j][0][:1]
                    if DASH.match(between) and a and b and a[0] in order and b[0] in order and order[b[0]] > order[a[0]] + 1:
                        rids.update(r['id'] for r in refs[order[a[0]] + 1:order[b[0]]])
            clean = re.sub(r'\s+', ' ', MARK.sub('', sent)).strip()
            if 30 <= len(clean) <= 1500:
                for rid in rids:
                    lst = out.setdefault(rid, [])
                    if clean not in lst: lst.append(clean)
    return out


def parse(xml_text):
    """-> (refs, sentences_by_ref_id). Raises JatsUnavailable when the XML has no reference list or no body citations."""
    try:
        root = etree.fromstring(xml_text.encode('utf-8'), parser=etree.XMLParser(recover=True, huge_tree=True))
    except etree.XMLSyntaxError as e:
        raise JatsUnavailable(f'jats_xml_error: {e}')
    for el in root.iter():   # drop namespaces (mml:, xlink:) so tag names are plain
        if isinstance(el.tag, str) and '}' in el.tag: el.tag = el.tag.split('}', 1)[1]
    refs = references(root)
    if not refs:
        raise JatsUnavailable('jats_no_reference_list')
    sents = citing_sentences(root, refs)
    if not sents:
        raise JatsUnavailable('jats_no_body_citations')
    return refs, sents
