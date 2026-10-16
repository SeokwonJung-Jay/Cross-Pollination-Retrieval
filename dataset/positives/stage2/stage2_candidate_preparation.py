#!/usr/bin/env python3
"""Stage 2: cross-topic candidate filtering + citation verification/context extraction.

  * candidate: q cites d, d has an abstract, and the OpenAlex topic sets (rank 1..3, --tags) of q and d are disjoint;
  * verified: bibliography title match >= 0.8, with no author-surname or year (+/-2) contradiction;
  * context: up to six citing sentences from the first full-text source that has q (Europe PMC / PMC JATS, then s2orc_v2,
    then the open-access PDF); a pair without a verified link and a citing sentence is dropped.

Cheap metadata filtering happens before any PDF is opened. Each q PDF is acquired and parsed
at most once, regardless of how many d candidates it has.

Inputs
------
--papers: JSON list or id-keyed object. IDs may be in `paperId` or `id`.
--references: JSON object {q_id: [reference, ...]}, JSON list, or JSONL. A reference uses the
  Semantic Scholar shape {citedPaper: {...}, contexts: [...], intents: [...]}; flat records with
  base_id/target_id are also accepted and completed from --papers.
--tags: Stage 1 JSONL output containing topics rank 1..3.

Outputs
-------
candidate_pairs.jsonl: every pair surviving the cheap filter.
pair_audit.jsonl: every candidate's verification/context outcome, including failures.
prepared_pairs.json: verified pairs in the next-stage input schema.
bases.jsonl: one processing summary per q.
summary.json: counts and provenance.
"""
from __future__ import annotations

import argparse
import os
import collections
import json
import re
import subprocess
import traceback
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))   # helper modules next to this file (also under python -I)
import _citation_context as ec
import _reference_verification as vr
import _jats
import _s2orc
from _fulltext import FullTextClient, FullTextUnavailable


def read_json_or_jsonl(path: Path):
    text = path.read_text()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return [json.loads(line) for line in text.splitlines() if line.strip()]


def paper_id(p):
    return p.get('paperId') or p.get('id')


def normalize_paper(p):
    p = dict(p)
    pid = paper_id(p)
    if not pid:
        raise ValueError('paper without paperId/id')
    p['paperId'] = pid
    return p


def load_papers(path):
    raw = read_json_or_jsonl(path)
    values = raw.values() if isinstance(raw, dict) else raw
    return {paper_id(p): normalize_paper(p) for p in values}


def load_tags(path):
    rows = read_json_or_jsonl(path)
    values = rows.values() if isinstance(rows, dict) else rows
    return {r['id']: r for r in values}


def load_references(path, papers):
    raw = read_json_or_jsonl(path)
    grouped = collections.defaultdict(list)
    if isinstance(raw, dict):
        iterable = ((qid, r) for qid, refs in raw.items() for r in refs)
    else:
        iterable = ((r.get('base_id') or r.get('citingPaperId') or r.get('q_id'), r) for r in raw)
    for qid, r0 in iterable:
        if not qid:
            raise ValueError('reference without base_id/citingPaperId/q_id')
        r = dict(r0)
        cited = dict(r.get('citedPaper') or {})
        did = cited.get('paperId') or r.get('target_id') or r.get('citedPaperId') or r.get('d_id')
        if not did:
            continue
        if did in papers:
            cited = {**papers[did], **cited}
        cited['paperId'] = did
        r['citedPaper'] = cited
        grouped[qid].append(r)
    return grouped


def topic_ids(tag):
    return {str(t.get('id')) for t in (tag or {}).get('topics') or [] if t.get('id')}


def topic_names(tag):
    return [t.get('name') for t in (tag or {}).get('topics') or [] if t.get('name')]


def s2_fields(paper):
    return sorted({x['category'] for x in paper.get('s2FieldsOfStudy') or [] if x.get('category')})


def append_jsonl(path, row):
    with path.open('a') as f:
        f.write(json.dumps(row, ensure_ascii=False) + '\n')


def write_json(path, value):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temp.replace(path)


def candidate_record(q, d, qt, dt):
    return {
        'base_id': q['paperId'], 'target_id': d['paperId'],
        'q_topics': topic_names(qt), 'd_topics': topic_names(dt),
        'q_topic_ids': sorted(topic_ids(qt)), 'd_topic_ids': sorted(topic_ids(dt)),
        'topic_sets_disjoint': True,
    }


def locate_entries(entries_raw, style):
    locs = []
    for i, raw in enumerate(entries_raw, 1):
        parsed = vr.parse_entry(raw)
        if style == 'numeric':
            match = re.match(r'\[(\d{1,3})\]', raw)
            locs.append(('num', int(match.group(1)) if match else i, parsed))
        elif style == 'lncs':
            match = re.match(r'(\d{1,3})\.', raw)
            locs.append(('num', int(match.group(1)) if match else i, parsed))
        else:
            locs.append(('ay', (ec.first_surname(raw, style), parsed['year'] if parsed else None), parsed))
    return locs


def verify_candidate(candidate, entries, entries_raw, style, locs, sentences, paragraphs):
    ref, d = candidate['reference'], candidate['d']
    best = max(((vr.s2orc_sim(d['title'], e['title']), e) for e in entries), key=lambda x: x[0], default=(0.0, None))
    cont = max(((vr.containment(d['title'], e['raw_full']), e) for e in entries), key=lambda x: x[0], default=(0.0, None))
    cont_ok = cont[0] >= 0.8 and cont[1] is not None and (
        not cont[1]['year'] or not d.get('year') or abs(cont[1]['year'] - d['year']) <= 2
    )
    title_ok = best[0] >= 0.8 or cont_ok
    matched = best[1] if best[0] >= 0.8 else (cont[1] if cont_ok else None)
    author_ok, year_ok = vr.author_year_match(matched, d) if title_ok else (None, None)
    status = vr.link_status(title_ok, author_ok, year_ok)

    ours, windows = [], []
    if status == 'verified':
        # find the matched entry by its full text; title+year was ambiguous when the title parse failed
        # (IEEE: several "Decis. Support Syst., vol" titles in one year sent the sentence lookup to another entry's number)
        loc = next((x for x in locs if x[2] and x[2]['raw_full'] == matched['raw_full']), None)
        if loc and loc[0] == 'num':
            sup = ec.citation_mode(sentences) != 'bracket'   # no superscript reading in [n]-style papers
            ours = [s for s in sentences if loc[1] in ec.numeric_ids(s, max_id=len(entries_raw), allow_superscript=sup)]
            windows = ec.paragraph_windows(paragraphs, lambda p: loc[1] in ec.numeric_ids(p, max_id=len(entries_raw), allow_superscript=sup))
        elif loc:
            surname, year = ec.locator_for_title(matched['raw_full'], d['title'], style)
            if surname and year:
                pattern = ec.author_year_pattern(surname, year)
                ours = [s for s in sentences if ec.cites(pattern, s)]
                windows = ec.paragraph_windows(paragraphs, lambda p: ec.cites(pattern, p))

    s2_contexts = ref.get('contexts') or []
    contexts = ours[:6] or s2_contexts
    source = 'pdf' if ours else ('semantic_scholar' if s2_contexts else 'none')
    return {
        'link_status': status,
        'title_sim': round(max(best[0], cont[0]), 3),
        'author_match': author_ok,
        'year_match': year_ok,
        'matched_reference': ({'title': matched['title'], 'year': matched['year']} if matched else None),
        'context_source': source,
        'contexts': contexts,
        'relevance': '\n'.join(contexts) if contexts else 'N/A',
        'windows': windows,
        'intents': ref.get('intents') or [],
    }


def jats_candidate(candidate, refs, sents):
    return structured_candidate(candidate, refs, sents, 'jats')


def structured_candidate(candidate, refs, sents, source_name):
    """Link d to a structured reference (JATS / s2orc_v2): Semantic Scholar matched id, DOI or PMID identity first, else the same title rule as the PDF path (title >= 0.8 or
    containment >= 0.8 with year +/-2), then the same author/year contradiction check. Contexts = the XML-linked citing sentences."""
    ref, d = candidate['reference'], candidate['d']
    ext = d.get('externalIds') or {}
    doi, pmid = str(ext.get('DOI') or '').lower(), str(ext.get('PubMed') or '')
    method, matched = None, None
    mid = next((r for r in refs if r.get('matched_paper_id') and str(r['matched_paper_id']) == str(d['paperId'])), None)
    if mid is not None:
        matched, method = mid, 'matched_paper_id'
    if matched is None and doi:
        matched = next((r for r in refs if r.get('doi') and r['doi'] == doi), None); method = 'doi' if matched else None
    if matched is None and pmid:
        matched = next((r for r in refs if r.get('pmid') and r['pmid'] == pmid), None); method = 'pmid' if matched else None
    best = max(((vr.s2orc_sim(d['title'], r['title']), r) for r in refs if r['title']), key=lambda x: x[0], default=(0.0, None))
    cont = max(((vr.containment(d['title'], r['raw_full']), r) for r in refs), key=lambda x: x[0], default=(0.0, None))
    if matched is None:
        if best[0] >= 0.8:
            matched, method = best[1], 'title'
        elif cont[0] >= 0.8 and cont[1] is not None and (not cont[1]['year'] or not d.get('year') or abs(cont[1]['year'] - d['year']) <= 2):
            matched, method = cont[1], 'title_containment'
    entry = {'raw_full': matched['raw_full'], 'year': matched['year']} if matched else None
    author_ok, year_ok = vr.author_year_match(entry, d) if matched else (None, None)
    status = vr.link_status(matched is not None, author_ok, year_ok)
    ours = sents.get(matched['id'], [])[:6] if (matched and status == 'verified') else []
    s2_contexts = ref.get('contexts') or []
    contexts = ours or s2_contexts
    source = source_name if ours else ('semantic_scholar' if s2_contexts else 'none')
    return {
        'link_status': status, 'link_method': method,
        'title_sim': round(max(best[0], cont[0]), 3),
        'author_match': author_ok, 'year_match': year_ok,
        'matched_reference': ({'title': matched['title'], 'year': matched['year']} if matched else None),
        'context_source': source, 'contexts': contexts,
        'relevance': '\n'.join(contexts) if contexts else 'N/A',
        'windows': [], 'intents': ref.get('intents') or [],
    }


def prepared_record(pair, q, d, qtag, dtag, prefix):
    return {
        'pair_id': f'{prefix}_{q["paperId"]}_{d["paperId"]}',   # <prefix>_<q corpusid>_<d corpusid>
        'base_id': q['paperId'], 'target_id': d['paperId'],
        'q': {
            'title': q['title'], 'abstract': q.get('abstract'), 'year': q.get('year'),
            'fields': topic_names(qtag)[:1], 'topics': topic_names(qtag),
            'subfield1': ((qtag.get('subfields') or [{}])[0].get('name')),
            's2_fields': s2_fields(q), 'citing_sentences': pair['contexts'], 'citing_windows': [],
        },
        'd': {
            'title': d['title'], 'abstract': d.get('abstract'), 'year': d.get('year'),
            'fields': topic_names(dtag)[:1], 'topics': topic_names(dtag),
            'subfield1': ((dtag.get('subfields') or [{}])[0].get('name')),
            's2_fields': s2_fields(d), 'paper_id': d['paperId'],
        },
        'intents': pair['intents'], 'context_source': pair['context_source'],
        'title_sim': pair['title_sim'], 'author_match': pair['author_match'], 'year_match': pair['year_match'],
        'topic_sets_disjoint': True, 'label_level': 'openalex_topic_rank1_3_disjoint',
        'context_version': 'v1-sentences', 'relevance': pair['relevance'],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--papers', type=Path, required=True)
    parser.add_argument('--references', type=Path, required=True)
    parser.add_argument('--tags', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--pdf-dir', type=Path)
    parser.add_argument('--fulltext-cache', type=Path, required=True)
    parser.add_argument('--pair-prefix', default='pair')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--jats-cache', type=Path,
                        help='Europe PMC JATS cache dir; when set, a q with a PMCID uses its JATS XML first and the PDF only as fallback')
    parser.add_argument('--s2orc-dir', type=Path,
                        help='directory of s2orc_v2 records named {q paperId}.json (second source after JATS)')
    parser.add_argument('--max-pdf-mb', type=float, default=25,
                        help='largest q PDF to download or read (default 25 MB)')
    args = parser.parse_args()

    papers, tags = load_papers(args.papers), load_tags(args.tags)
    references = load_references(args.references, papers)
    args.out.mkdir(parents=True, exist_ok=True)
    candidate_path, audit_path, base_path = (args.out / n for n in ('candidate_pairs.jsonl', 'pair_audit.jsonl', 'bases.jsonl'))
    if not args.resume:
        for path in (candidate_path, audit_path, base_path):
            if path.exists():
                raise SystemExit(f'{path} exists; use a new --out or --resume')
    done = set()
    if args.resume and base_path.exists():
        done = {json.loads(line)['base_id'] for line in base_path.open() if line.strip()}

    candidates = collections.defaultdict(list)
    seen = set()
    for qid, refs in references.items():
        q, qtag = papers.get(qid), tags.get(qid)
        if not q or not q.get('abstract') or not topic_ids(qtag):
            continue
        for ref in refs:
            d = normalize_paper(ref['citedPaper'])
            did, dtag = d['paperId'], tags.get(d['paperId'])
            key = (qid, did)
            if key in seen or not d.get('title') or not d.get('abstract') or not topic_ids(dtag):
                continue
            seen.add(key)
            if topic_ids(qtag) & topic_ids(dtag):
                continue
            record = candidate_record(q, d, qtag, dtag)
            candidates[qid].append({'q': q, 'd': d, 'qtag': qtag, 'dtag': dtag, 'reference': ref, 'candidate': record})

    if not args.resume:   # candidates are deterministic: written once
        with candidate_path.open('w') as f:
            f.writelines(json.dumps(x['candidate'], ensure_ascii=False) + '\n' for v in candidates.values() for x in v)

    # per candidate, sources in order Europe PMC JATS -> s2orc_v2 -> PDF rule parser. A pair is
    # prepared only when a source both verifies the link and yields >= 1 citing sentence; pairs without a citing sentence are
    # dropped (Semantic Scholar contexts are not used). When JATS exists it is final: no fallback if it does not link d.
    fulltext = FullTextClient(args.fulltext_cache, args.pdf_dir, max_bytes=int(args.max_pdf_mb * 1024 * 1024))
    prepared, status_counts, context_counts = [], collections.Counter(), collections.Counter()
    for qid in sorted(candidates):
        if qid in done:
            continue
        group, q = candidates[qid], candidates[qid][0]['q']
        summary = {'base_id': qid, 'n_candidates': len(group), 'sources': {}}
        cache = {}

        def jats():
            if 'jats' not in cache:
                cache['jats'] = None
                pmcid = (q.get('externalIds') or {}).get('PubMedCentral') or q.get('pmcid')
                if args.jats_cache and pmcid:
                    try:
                        cache['jats'] = _jats.parse(_jats.fetch(pmcid, args.jats_cache))
                        summary['sources']['jats'] = 'ok'
                    except Exception as error:
                        summary['sources']['jats'] = f'{type(error).__name__}: {error}'[:200]
            return cache['jats']

        def s2orc():
            if 's2orc' not in cache:
                cache['s2orc'] = None
                f = args.s2orc_dir / f'{qid}.json' if args.s2orc_dir else None
                if f is not None and f.exists():
                    try:
                        cache['s2orc'] = _s2orc.parse(f.read_text())
                        summary['sources']['s2orc'] = 'ok'
                    except Exception as error:
                        summary['sources']['s2orc'] = f'{type(error).__name__}: {error}'[:200]
                elif args.s2orc_dir:
                    summary['sources']['s2orc'] = 'no_record'
            return cache['s2orc']

        def pdf():
            if 'pdf' not in cache:
                cache['pdf'] = None
                try:
                    document = fulltext.get(q)
                    entries_raw, style = vr.reference_entries(document['cached_pdf'])
                    if entries_raw is None:
                        raise FullTextUnavailable('no_references_section')
                    entries = [x for x in (vr.parse_entry(raw) for raw in entries_raw) if x]
                    ratio = len(entries) / max(len(references[qid]), 1)
                    sentences = ec.body_sentences(document['cached_pdf']) if ratio >= 0.5 else []
                    paragraphs = ec.body_paragraphs(document['cached_pdf']) if ratio >= 0.5 else []
                    cache['pdf'] = (entries, entries_raw, style, locate_entries(entries_raw, style), sentences, paragraphs, ratio < 0.5)
                    summary['sources']['pdf'] = f'ok style={style} entries={len(entries)} ratio={ratio:.2f}'
                except Exception as error:
                    summary['sources']['pdf'] = f'{type(error).__name__}: {error}'[:200]
            return cache['pdf']

        rows = []   # one write per q
        for item in group:
            tried, final = [], None
            j = jats()
            if j is not None:
                r = structured_candidate(item, j[0], j[1], 'jats'); tried.append(('jats', r['link_status'], r['context_source'] == 'jats'))
                final = r if (r['link_status'] == 'verified' and r['context_source'] == 'jats') else None
                decision = 'prepared' if final else ('jats_no_sentence' if r['link_status'] == 'verified' else 'jats_not_linked')
            else:
                for name in ('s2orc', 'pdf'):
                    if name == 's2orc':
                        s = s2orc()
                        if s is None: continue
                        r = structured_candidate(item, s[0], s[1], 's2orc')
                    else:
                        pd = pdf()
                        if pd is None: continue
                        r = verify_candidate(item, *pd[:6])
                        if r['link_status'] == 'unverified' and pd[6]:
                            r['link_status'] = 'unverifiable_parse_suspect'
                    tried.append((name, r['link_status'], r['context_source'] == name))
                    if r['link_status'] == 'verified' and r['context_source'] == name:
                        final = r; break
                decision = 'prepared' if final else ('no_source' if not tried else
                           'no_sentence' if any(t[1] == 'verified' for t in tried) else 'not_linked')
            row = {**item['candidate'], 'decision': decision, 'tried': tried}
            if final:
                row.update(final); context_counts[final['context_source']] += 1
                prepared.append(prepared_record(final, q, item['d'], item['qtag'], item['dtag'], args.pair_prefix))
            rows.append(row)
            status_counts[decision] += 1
        with audit_path.open('a') as f:
            f.writelines(json.dumps(r, ensure_ascii=False) + '\n' for r in rows)
        summary['n_verified'] = sum(1 for x in prepared if x['base_id'] == qid)
        append_jsonl(base_path, summary)

    if args.resume and (args.out / 'prepared_pairs.json').exists():
        old = json.loads((args.out / 'prepared_pairs.json').read_text())
        by_id = {x['pair_id']: x for x in old}
        by_id.update({x['pair_id']: x for x in prepared})
        prepared = list(by_id.values())
    write_json(args.out / 'prepared_pairs.json', prepared)
    commit = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip() or 'unknown'
    summary = {'pipeline': 'stage2_candidate_preparation', 'source_commit': commit,
               'candidate_pairs': sum(len(x) for x in candidates.values()), 'candidate_q': len(candidates),
               'prepared_pairs': len(prepared), 'prepared_q': len({x['base_id'] for x in prepared}),
               'link_status': dict(status_counts), 'context_source': dict(context_counts),
               'criteria': {'topic': 'rank-1..3 sets disjoint', 'title_similarity': 0.8,
                            'author': 'no surname contradiction', 'year_tolerance': 2,
                            'context_priority': ['jats (Europe PMC; final when present)', 's2orc_v2', 'pdf rule parser', 'drop: no citing sentence']}}
    write_json(args.out / 'summary.json', summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
