"""Fetch a cited paper's public PDF and extract every page without truncation."""
from __future__ import annotations

import hashlib
import ipaddress
import json
from pathlib import Path
import re
import socket
import time
from urllib.parse import urljoin, urlparse


class FullTextUnavailable(RuntimeError):
    """Expected acquisition/extraction failure; never fall back to the abstract."""


def validate_public_url(url):
    parsed = urlparse(url)
    if parsed.scheme not in {'http', 'https'} or not parsed.hostname or parsed.username or parsed.password:
        raise FullTextUnavailable('unsupported_pdf_url')
    try:
        addresses = socket.getaddrinfo(parsed.hostname, parsed.port or (443 if parsed.scheme == 'https' else 80))
    except OSError:
        raise FullTextUnavailable('pdf_host_unreachable') from None
    if not addresses or any(not ipaddress.ip_address(a[4][0]).is_global for a in addresses):
        raise FullTextUnavailable('non_public_pdf_url')


def extract_pdf(data):
    try:
        import pymupdf
    except ImportError:
        raise RuntimeError('Install pymupdf (dataset/requirements.txt) for PDF support') from None
    try:
        with pymupdf.open(stream=data, filetype='pdf') as doc:
            if doc.needs_pass or len(doc) == 0:
                raise FullTextUnavailable('encrypted_or_empty_pdf')
            pages = [page.get_text('text', sort=True).strip() for page in doc]
    except FullTextUnavailable:
        raise
    except Exception:
        raise FullTextUnavailable('pdf_extraction_failed') from None
    empty = [i + 1 for i, text in enumerate(pages) if not text]
    if empty:
        # Could be a scan, a figure-only page or a genuinely blank page. Do not
        # silently claim the whole document is available when this is unknown.
        raise FullTextUnavailable('textless_pages_require_ocr_or_review:' + ','.join(map(str, empty)))
    if sum(len(text) for text in pages) < 500:
        raise FullTextUnavailable('too_little_extracted_text')
    return {'text': '\n\n'.join(f'[Page {i}]\n{text}' for i, text in enumerate(pages, 1)),
            'page_count': len(pages), 'pages_extracted': len(pages),
            'extraction': 'pymupdf_text_all_pages', 'truncated': False,
            'limitations': 'Text only: figures, scanned text, equations and table layout may not be faithfully represented.'}


class FullTextClient:
    def __init__(self, cache, local_dir=None, max_bytes=25 * 1024 * 1024):
        import requests
        # Separate session: do not forward the S2 API key to PDF hosts.
        self.session = requests.Session()
        self.session.headers['User-Agent'] = 'cpr-dataset-research/1.0'
        self.cache = Path(cache)
        self.local_dir = Path(local_dir) if local_dir else None
        self.max_bytes = max_bytes

    def download(self, url):
        import requests
        for _ in range(6):
            validate_public_url(url)
            try:
                with self.session.get(url, stream=True, allow_redirects=False, timeout=(10, 30)) as response:
                    if response.status_code in {301, 302, 303, 307, 308}:
                        location = response.headers.get('Location')
                        if not location:
                            raise FullTextUnavailable('pdf_redirect_without_location')
                        url = urljoin(url, location)
                        continue
                    if response.status_code != 200:
                        raise FullTextUnavailable(f'pdf_http_{response.status_code}')
                    data, started = bytearray(), time.monotonic()
                    for chunk in response.iter_content(65536):
                        if time.monotonic() - started > 90:
                            raise FullTextUnavailable('pdf_download_time_limit')
                        data.extend(chunk)
                        if len(data) > self.max_bytes:
                            raise FullTextUnavailable('pdf_exceeds_download_size_limit')
                    if b'%PDF-' not in data[:1024]:
                        raise FullTextUnavailable('url_did_not_return_pdf')
                    return bytes(data), url
            except requests.RequestException:
                raise FullTextUnavailable('pdf_network_error') from None
        raise FullTextUnavailable('too_many_pdf_redirects')

    def get(self, paper):
        pid = paper.get('paperId') or ''
        if not re.fullmatch(r'[A-Za-z0-9_-]+', pid):
            raise FullTextUnavailable('invalid_paper_id')
        candidates = []
        if self.local_dir:
            local = self.local_dir / (pid + '.pdf')
            if local.is_file():
                candidates.append(('local', str(local)))
        oa = (paper.get('openAccessPdf') or {}).get('url')
        if oa:
            candidates.append(('url', oa))
        arxiv = str((paper.get('externalIds') or {}).get('ArXiv') or '')
        if re.fullmatch(r'(?:\d{4}\.\d{4,5}|[A-Za-z.-]+/\d{7})(?:v\d+)?', arxiv):
            candidates.append(('url', 'https://arxiv.org/pdf/' + arxiv))
        if not candidates:
            raise FullTextUnavailable('no_open_pdf_url_or_local_pdf')
        failures = []
        for kind, location in dict.fromkeys(candidates):
            source = [pid, kind, location]
            if kind == 'local':
                stat = Path(location).stat()
                source += [stat.st_size, stat.st_mtime_ns]
            key = hashlib.sha256(json.dumps(source).encode()).hexdigest()
            destination = self.cache / (key + '.json')
            if destination.exists():
                return json.loads(destination.read_text())
            try:
                if kind == 'local':
                    if Path(location).stat().st_size > self.max_bytes:
                        raise FullTextUnavailable('local_pdf_exceeds_size_limit')
                    data, resolved = Path(location).read_bytes(), location
                else:
                    data, resolved = self.download(location)
                result = {**extract_pdf(data), 'paper_id': pid, 'source': resolved,
                          'source_kind': kind, 'sha256': hashlib.sha256(data).hexdigest()}
                self.cache.mkdir(parents=True, exist_ok=True)
                pdf_path = self.cache / (key + '.pdf')
                pdf_path.write_bytes(data)
                result['cached_pdf'] = str(pdf_path.resolve())
                temp = destination.with_suffix('.tmp')
                temp.write_text(json.dumps(result, ensure_ascii=False) + '\n')
                temp.replace(destination)
                return result
            except FullTextUnavailable as error:
                failures.append(str(error))
        raise FullTextUnavailable('; '.join(failures))
