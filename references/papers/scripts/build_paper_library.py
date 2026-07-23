#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[2]
PAPERS_DIR = ROOT / "papers"
LISTS_DIR = PAPERS_DIR / "sources"
LIBRARY_DIR = PAPERS_DIR / "knowledge_base"
PDF_DIR = LIBRARY_DIR / "active"
ORGANIZED_DIR = PAPERS_DIR / "legacy_organized_views"
MANIFEST_PATH = LIBRARY_DIR / "manifest.json"
REPORT_PATH = LIBRARY_DIR / "report.md"
OVERRIDES_PATH = PAPERS_DIR / "sources" / "manual_source_overrides.json"

LIST_SPECS = {
    "arxiv_paper_list.md": {"id_re": re.compile(r"^\d+$"), "style": "arxiv"},
    "proceedings_paper_list.md": {"id_re": re.compile(r"^P\d+$"), "style": "proceedings"},
    "FINAL_curated_paper_list.md": {"id_re": re.compile(r"^[A-Z]\d+$"), "style": "final"},
}

USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)
REQUEST_HEADERS = {"User-Agent": USER_AGENT}
NETWORK_DELAY_SECONDS = 0.2

TITLE_ALIAS_TO_ARXIV = {
    "forecastinggpuperformancefordeeplearningtrainingandinferenceneusight": "2407.13853",
    "forecastingllminferenceperformanceviahardwareagnosticanalyticalmodelinglife": "2508.00904",
    "understandingandoptimizingmultistageaiinferencepipelinesmist": "2504.09775",
    "neuroscalaradeeplearningframeworkforfastaccuratecyclelevelperformanceprediction": "2509.22410",
    "phantoramaximizingcodereuseinsimulationbasedmachinelearningsystemperformanceestimation": "2505.01616",
}


@dataclass
class Occurrence:
    source_list: str
    section: str
    subsection: str
    paper_id: str
    title: str
    authors: str
    year: str
    venue: str
    extra: str
    links: list[str]
    arxiv_id: str | None


@dataclass
class PaperRecord:
    canonical_id: str
    title: str
    authors: str
    year: str
    venue: str
    arxiv_id: str | None
    source_status: str
    source_url: str | None
    source_kind: str | None
    local_pdf: str | None
    occurrences: list[dict]


def normalize_title(title: str) -> str:
    title = title.lower()
    title = re.sub(r"\([^)]*\)", "", title)
    title = title.replace("&", "and")
    title = re.sub(r"[^a-z0-9]+", "", title)
    return title


def slugify(text: str, max_len: int = 96) -> str:
    text = text.lower()
    text = text.replace("&", "and")
    text = re.sub(r"[^a-z0-9]+", "_", text).strip("_")
    return text[:max_len].rstrip("_") or "paper"


def extract_links(cell: str) -> list[str]:
    return re.findall(r"\[[^\]]+\]\(([^)]+)\)", cell)


def extract_arxiv_id(text: str) -> str | None:
    if not text:
        return None
    match = re.search(r"(\d{4}\.\d{4,5})(?:v\d+)?", text)
    if match:
        return match.group(1)
    return None


def load_overrides() -> dict:
    if not OVERRIDES_PATH.exists():
        return {"papers": {}, "aliases": {}}
    data = json.loads(OVERRIDES_PATH.read_text())
    data.setdefault("papers", {})
    data.setdefault("aliases", {})
    return data


def parse_markdown_list(path: Path) -> list[Occurrence]:
    spec = LIST_SPECS[path.name]
    section = ""
    subsection = ""
    occurrences: list[Occurrence] = []

    for raw_line in path.read_text().splitlines():
        line = raw_line.rstrip()
        if line.startswith("## "):
            section = line[3:].strip()
            subsection = ""
            continue
        if line.startswith("### "):
            subsection = line[4:].strip()
            continue
        if not line.startswith("|") or line.startswith("|---") or "Title" in line:
            continue

        cells = [cell.strip() for cell in line.split("|")[1:-1]]
        if not cells or not spec["id_re"].fullmatch(cells[0]):
            continue

        links: list[str] = []
        arxiv_id = None
        while len(cells) < 6:
            cells.append("")

        if spec["style"] == "arxiv":
            paper_id, title, authors, year, arxiv_cell, venue = cells[:6]
            links.extend(extract_links(arxiv_cell))
            arxiv_id = extract_arxiv_id(arxiv_cell)
            extra = arxiv_cell
        elif spec["style"] == "proceedings":
            paper_id, title, authors, venue, year, notes = cells[:6]
            links.extend(extract_links(notes))
            extra = notes
        else:
            paper_id, title, authors, venue, year, links_cell = cells[:6]
            links.extend(extract_links(links_cell))
            arxiv_id = extract_arxiv_id(links_cell)
            extra = links_cell

        normalized = normalize_title(title)
        if not arxiv_id and normalized in TITLE_ALIAS_TO_ARXIV:
            arxiv_id = TITLE_ALIAS_TO_ARXIV[normalized]

        occurrences.append(
            Occurrence(
                source_list=path.name,
                section=section,
                subsection=subsection,
                paper_id=paper_id,
                title=title,
                authors=authors,
                year=year,
                venue=venue,
                extra=extra,
                links=links,
                arxiv_id=arxiv_id,
            )
        )

    return occurrences


def canonical_key(occurrence: Occurrence, overrides: dict) -> str:
    normalized = normalize_title(occurrence.title)
    alias = overrides["aliases"].get(normalized)
    if alias:
        return alias
    if occurrence.arxiv_id:
        return f"arxiv:{occurrence.arxiv_id}"
    return f"title:{normalized}"


def build_records(overrides: dict) -> list[PaperRecord]:
    grouped: dict[str, list[Occurrence]] = {}
    for list_name in LIST_SPECS:
        for occurrence in parse_markdown_list(LISTS_DIR / list_name):
            grouped.setdefault(canonical_key(occurrence, overrides), []).append(occurrence)

    records: list[PaperRecord] = []
    for key, occurrences in sorted(grouped.items(), key=lambda item: item[1][0].title.lower()):
        occurrences.sort(key=lambda occ: (occ.source_list, occ.paper_id))
        primary = choose_primary_occurrence(occurrences)
        normalized = normalize_title(primary.title)
        record = PaperRecord(
            canonical_id=f"{primary.year or 'unknown'}__{slugify(primary.title)}",
            title=primary.title,
            authors=primary.authors,
            year=primary.year,
            venue=primary.venue,
            arxiv_id=primary.arxiv_id,
            source_status="unresolved",
            source_url=None,
            source_kind=None,
            local_pdf=None,
            occurrences=[asdict(occ) for occ in occurrences],
        )
        maybe_override = overrides["papers"].get(normalized)
        if maybe_override:
            apply_override(record, maybe_override)
        records.append(record)
    return records


def choose_primary_occurrence(occurrences: list[Occurrence]) -> Occurrence:
    scored: list[tuple[int, Occurrence]] = []
    for occ in occurrences:
        score = 0
        if occ.arxiv_id:
            score += 40
        if occ.source_list == "FINAL_curated_paper_list.md":
            score += 20
        elif occ.source_list == "arxiv_paper_list.md":
            score += 10
        if occ.links:
            score += 5
        if occ.authors and occ.authors != "—":
            score += 2
        scored.append((score, occ))
    return max(scored, key=lambda item: item[0])[1]


def apply_override(record: PaperRecord, override: dict) -> None:
    local_path = override.get("local_path")
    if local_path:
        record.source_status = "resolved"
        record.source_kind = "local"
        record.local_pdf = str((ROOT / local_path).resolve())
        record.source_url = override.get("source_url")
        return
    pdf_url = override.get("pdf_url")
    if pdf_url:
        record.source_status = "candidate"
        record.source_kind = "manual_pdf_url"
        record.source_url = pdf_url
        return
    landing_page = override.get("landing_page")
    if landing_page:
        record.source_status = "candidate"
        record.source_kind = "manual_landing_page"
        record.source_url = landing_page


def request_json(url: str) -> dict | None:
    req = urllib.request.Request(url, headers=REQUEST_HEADERS)
    with urllib.request.urlopen(req, timeout=30) as resp:
        payload = resp.read()
    time.sleep(NETWORK_DELAY_SECONDS)
    return json.loads(payload)


def request_bytes(url: str) -> tuple[bytes, str]:
    req = urllib.request.Request(url, headers=REQUEST_HEADERS)
    with urllib.request.urlopen(req, timeout=60) as resp:
        body = resp.read()
        content_type = resp.headers.get("Content-Type", "")
        final_url = resp.geturl()
    time.sleep(NETWORK_DELAY_SECONDS)
    return body, final_url or content_type


def resolve_record(record: PaperRecord) -> None:
    if record.local_pdf and Path(record.local_pdf).exists():
        record.source_status = "resolved"
        record.source_kind = record.source_kind or "local"
        return

    candidates = candidate_urls(record)
    resolved = None
    for candidate_kind, candidate_url in candidates:
        resolved = resolve_candidate(candidate_kind, candidate_url)
        if resolved:
            record.source_kind, record.source_url = resolved
            record.source_status = "candidate"
            return


def candidate_urls(record: PaperRecord) -> list[tuple[str, str]]:
    urls: list[tuple[str, str]] = []
    if record.source_url:
        urls.append((record.source_kind or "manual", record.source_url))
    if record.arxiv_id:
        urls.append(("arxiv_pdf", f"https://arxiv.org/pdf/{record.arxiv_id}.pdf"))
    for occurrence in record.occurrences:
        for link in occurrence.get("links", []):
            kind = "direct_pdf" if link.lower().endswith(".pdf") else "landing_page"
            urls.append((kind, link))
    urls.extend(discover_open_access_candidates(record))

    seen = set()
    deduped = []
    for kind, url in urls:
        if not url or url in seen:
            continue
        seen.add(url)
        deduped.append((kind, url))
    return deduped


def discover_open_access_candidates(record: PaperRecord) -> list[tuple[str, str]]:
    if record.arxiv_id:
        return []

    candidates: list[tuple[str, str]] = []
    search_terms = urllib.parse.quote(record.title)
    openalex_url = f"https://api.openalex.org/works?search={search_terms}&per-page=5"
    try:
        payload = request_json(openalex_url)
    except Exception:
        payload = None
    if payload:
        for work in payload.get("results", []):
            work_title = work.get("display_name", "")
            if title_match_score(record.title, work_title) < 0.72:
                continue
            best_location = work.get("best_oa_location") or {}
            open_access = work.get("open_access") or {}
            for oa_url in [
                best_location.get("pdf_url"),
                best_location.get("landing_page_url"),
                open_access.get("oa_url"),
            ]:
                if not oa_url:
                    continue
                kind = "openalex_pdf" if oa_url.lower().endswith(".pdf") else "openalex_page"
                candidates.append((kind, oa_url))
            for location in work.get("locations") or []:
                pdf_url = location.get("pdf_url")
                landing_page = location.get("landing_page_url")
                if pdf_url:
                    candidates.append(("openalex_pdf", pdf_url))
                if landing_page:
                    candidates.append(("openalex_page", landing_page))

    s2_url = (
        "https://api.semanticscholar.org/graph/v1/paper/search/match?"
        f"query={search_terms}&fields=title,year,openAccessPdf,externalIds,url"
    )
    try:
        payload = request_json(s2_url)
    except Exception:
        payload = None
    if payload:
        if isinstance(payload, dict):
            paper = payload.get("data") or payload
        else:
            paper = payload
        if isinstance(paper, list):
            paper = paper[0] if paper else {}
        if not isinstance(paper, dict):
            paper = {}
        pdf = ((paper.get("openAccessPdf") or {}).get("url"))
        if pdf:
            kind = "semantic_scholar_pdf" if pdf.lower().endswith(".pdf") else "semantic_scholar_page"
            candidates.append((kind, pdf))
        external_ids = paper.get("externalIds") or {}
        arxiv = None
        if isinstance(external_ids, dict):
            arxiv = (
                external_ids.get("ArXiv")
                or external_ids.get("ARXIV")
                or external_ids.get("Arxiv")
            )
        if isinstance(arxiv, str) and extract_arxiv_id(arxiv):
            arxiv = extract_arxiv_id(arxiv)
            candidates.append(("semantic_scholar_arxiv", f"https://arxiv.org/pdf/{arxiv}.pdf"))
        paper_url = paper.get("url")
        if paper_url:
            candidates.append(("semantic_scholar_page", paper_url))

    return candidates


def title_match_score(left: str, right: str) -> float:
    left_norm = normalize_title(left)
    right_norm = normalize_title(right)
    if not left_norm or not right_norm:
        return 0.0
    if left_norm == right_norm:
        return 1.0
    left_tokens = split_tokens(left)
    right_tokens = split_tokens(right)
    if not left_tokens or not right_tokens:
        return 0.0
    overlap = len(left_tokens & right_tokens)
    return overlap / max(len(left_tokens), len(right_tokens))


def split_tokens(title: str) -> set[str]:
    title = title.lower()
    title = title.replace("&", "and")
    return {token for token in re.split(r"[^a-z0-9]+", title) if token}


def resolve_candidate(kind: str, url: str) -> tuple[str, str] | None:
    if url.lower().endswith(".pdf"):
        return kind, url
    if "arxiv.org/abs/" in url:
        arxiv_id = extract_arxiv_id(url)
        if arxiv_id:
            return "arxiv_pdf", f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    if "github.com" in url:
        return None
    pdf_url = discover_pdf_from_page(url)
    if pdf_url:
        return f"{kind}_discovered_pdf", pdf_url
    return None


def discover_pdf_from_page(url: str) -> str | None:
    try:
        req = urllib.request.Request(url, headers=REQUEST_HEADERS)
        with urllib.request.urlopen(req, timeout=30) as resp:
            content_type = resp.headers.get("Content-Type", "")
            body = resp.read()
            final_url = resp.geturl()
    except Exception:
        return None
    time.sleep(NETWORK_DELAY_SECONDS)
    if "pdf" in content_type.lower() or body.startswith(b"%PDF-"):
        return final_url

    html = body.decode("utf-8", errors="ignore")

    meta_match = re.search(r'citation_pdf_url"\s+content="([^"]+)"', html, flags=re.IGNORECASE)
    if meta_match:
        return urllib.parse.urljoin(final_url, meta_match.group(1))

    og_match = re.search(r'property="og:pdf" content="([^"]+)"', html, flags=re.IGNORECASE)
    if og_match:
        return urllib.parse.urljoin(final_url, og_match.group(1))

    for pattern in [
        r'href="([^"]+\.pdf(?:\?[^"]*)?)"',
        r"href='([^']+\.pdf(?:\?[^']*)?)'",
    ]:
        match = re.search(pattern, html, flags=re.IGNORECASE)
        if match:
            return urllib.parse.urljoin(final_url, match.group(1))
    return None


def download_record(record: PaperRecord) -> None:
    dest = PDF_DIR / f"{record.canonical_id}.pdf"
    if dest.exists():
        record.local_pdf = str(dest.resolve())
        record.source_status = "downloaded"
        return

    if record.local_pdf and Path(record.local_pdf).exists():
        if not dest.exists():
            shutil.copy2(record.local_pdf, dest)
        record.local_pdf = str(dest.resolve())
        record.source_status = "downloaded"
        return

    if not record.source_url:
        return

    tmp = dest.with_suffix(".tmp")
    req = urllib.request.Request(record.source_url, headers=REQUEST_HEADERS)
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            body = resp.read()
            final_url = resp.geturl()
            content_type = resp.headers.get("Content-Type", "")
    except Exception:
        return
    time.sleep(NETWORK_DELAY_SECONDS)

    if not is_pdf_payload(body, content_type):
        return

    tmp.write_bytes(body)
    tmp.replace(dest)
    record.local_pdf = str(dest.resolve())
    record.source_url = final_url
    record.source_status = "downloaded"


def is_pdf_payload(body: bytes, content_type: str) -> bool:
    if body.startswith(b"%PDF-"):
        return True
    if "application/pdf" in content_type.lower():
        return True
    return False


def build_organized_views(records: list[PaperRecord]) -> None:
    ORGANIZED_DIR.mkdir(parents=True, exist_ok=True)
    for record in records:
        if not record.local_pdf:
            continue
        source_pdf = Path(record.local_pdf)
        for occ in record.occurrences:
            list_dir = ORGANIZED_DIR / Path(occ["source_list"]).stem
            section_dir = list_dir / slugify(occ["section"] or "uncategorized", 72)
            if occ["subsection"]:
                section_dir = section_dir / slugify(occ["subsection"], 72)
            section_dir.mkdir(parents=True, exist_ok=True)
            file_name = f"{occ['paper_id']}__{slugify(record.title)}.pdf"
            link_path = section_dir / file_name
            ensure_link(link_path, source_pdf)

    for list_name in LIST_SPECS:
        build_index(Path(list_name).stem, records)


def ensure_link(link_path: Path, target: Path) -> None:
    if link_path.exists() or link_path.is_symlink():
        if link_path.is_symlink() and link_path.resolve() == target.resolve():
            return
        link_path.unlink()
    relative_target = os.path.relpath(target, link_path.parent)
    link_path.symlink_to(relative_target)


def build_index(list_stem: str, records: list[PaperRecord]) -> None:
    rows = []
    for record in records:
        for occ in record.occurrences:
            if Path(occ["source_list"]).stem != list_stem:
                continue
            rows.append((occ["section"], occ["subsection"], occ["paper_id"], record, occ))
    rows.sort(key=lambda item: order_key(item[2]))

    lines = [f"# {list_stem}", ""]
    current_section = None
    current_subsection = None
    for section, subsection, paper_id, record, occ in rows:
        if section != current_section:
            lines.extend([f"## {section}", ""])
            current_section = section
            current_subsection = None
        if subsection and subsection != current_subsection:
            lines.extend([f"### {subsection}", ""])
            current_subsection = subsection
        status = record.source_status
        local_ref = local_reference_for(list_stem, section, subsection, paper_id, record.title)
        source = record.source_url or "N/A"
        lines.append(
            f"- `{paper_id}` [{record.title}]({local_ref}) | {status} | source: {source}"
        )
    index_path = ORGANIZED_DIR / list_stem / "README.md"
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_text("\n".join(lines) + "\n")


def local_reference_for(list_stem: str, section: str, subsection: str, paper_id: str, title: str) -> str:
    parts = [slugify(section or "uncategorized", 72)]
    if subsection:
        parts.append(slugify(subsection, 72))
    parts.append(f"{paper_id}__{slugify(title)}.pdf")
    return "/".join(parts)


def order_key(identifier: str) -> tuple[int, str]:
    match = re.match(r"([A-Z]?)(\d+)", identifier)
    if not match:
        return (sys.maxsize, identifier)
    prefix, number = match.groups()
    prefix_score = 0
    if prefix:
        prefix_score = ord(prefix[0]) - ord("A") + 1
    return (prefix_score * 1000 + int(number), identifier)


def write_outputs(records: list[PaperRecord]) -> None:
    LIBRARY_DIR.mkdir(parents=True, exist_ok=True)
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps([asdict(record) for record in records], indent=2) + "\n")
    REPORT_PATH.write_text(build_report(records))


def build_report(records: list[PaperRecord]) -> str:
    total = len(records)
    available = sum(
        1 for record in records if record.local_pdf and Path(record.local_pdf).exists()
    )
    resolved_sources = sum(
        1 for record in records if record.source_url and not record.local_pdf
    )
    unresolved = [
        record
        for record in records
        if not record.source_url and not (record.local_pdf and Path(record.local_pdf).exists())
    ]
    lines = [
        "# Paper Knowledge Base Report",
        "",
        f"- Total canonical papers: {total}",
        f"- Available local PDFs: {available}",
        f"- Resolved remote PDF sources: {resolved_sources}",
        f"- Unresolved papers: {len(unresolved)}",
        "",
        "## Unresolved",
        "",
    ]
    if not unresolved:
        lines.append("- None")
    else:
        for record in unresolved:
            lines.append(f"- {record.title} ({record.year})")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a local paper library from markdown lists.")
    parser.add_argument("--resolve", action="store_true", help="Resolve download URLs via web APIs/pages.")
    parser.add_argument("--download", action="store_true", help="Download resolved PDFs.")
    parser.add_argument(
        "--organize",
        action="store_true",
        help="Create legacy per-list organized symlink views and README indexes.",
    )
    args = parser.parse_args()

    overrides = load_overrides()
    records = build_records(overrides)

    if args.resolve or args.download:
        for idx, record in enumerate(records, start=1):
            print(f"[resolve {idx}/{len(records)}] {record.title}", file=sys.stderr)
            if record.source_status == "resolved" and record.local_pdf:
                continue
            try:
                resolve_record(record)
            except urllib.error.HTTPError:
                continue
            except urllib.error.URLError:
                continue
            except TimeoutError:
                continue

    if args.download:
        PDF_DIR.mkdir(parents=True, exist_ok=True)
        for idx, record in enumerate(records, start=1):
            print(f"[download {idx}/{len(records)}] {record.title}", file=sys.stderr)
            try:
                download_record(record)
            except urllib.error.HTTPError:
                continue
            except urllib.error.URLError:
                continue
            except TimeoutError:
                continue

    if args.organize:
        build_organized_views(records)

    write_outputs(records)
    available = sum(1 for record in records if record.local_pdf and Path(record.local_pdf).exists())
    resolved = sum(1 for record in records if record.source_url)
    print(
        f"[summary] total={len(records)} available_local={available} resolved_sources={resolved}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
