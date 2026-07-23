#!/usr/bin/env python3
from __future__ import annotations

import json
import re
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
LIBRARY_DIR = ROOT / "papers" / "knowledge_base"
UNRESOLVED_DIR = LIBRARY_DIR / "inbox"
PDF_DIR = LIBRARY_DIR / "active"
MANIFEST_PATH = LIBRARY_DIR / "manifest.json"
REPORT_PATH = LIBRARY_DIR / "report.md"
TODAY = "2026-04-11"


def slugify(text: str, max_len: int = 96) -> str:
    text = text.lower()
    text = text.replace("&", "and")
    text = re.sub(r"[^a-z0-9]+", "_", text).strip("_")
    return text[:max_len].rstrip("_") or "paper"


RECOVERED_PAPERS = [
    {
        "filename": "baghsorkhi2010.pdf",
        "title": "An Adaptive Performance Modeling Tool for GPU Architectures",
        "authors": "S. S. Baghsorkhi, M. Delahaye, S. J. Patel, W. D. Gropp, W.-m. W. Hwu",
        "year": "2010",
        "venue": "PPoPP",
        "arxiv_id": None,
        "source_url": "https://doi.org/10.1145/1693453.1693470",
        "source_kind": "manual_local_pdf",
        "section": "Highest Priority",
        "paper_id": "MM10",
        "extra": "[DOI](https://doi.org/10.1145/1693453.1693470)",
        "links": ["https://doi.org/10.1145/1693453.1693470"],
    },
    {
        "filename": "2001.07104v3.pdf",
        "title": "A Simple Model for Portable and Fast Prediction of Execution Time and Power Consumption of GPU Kernels",
        "authors": "L. Braun, S. Nikas, C. Song, V. Heuveline, H. Froning",
        "year": "2021",
        "venue": "ACM TACO",
        "arxiv_id": "2001.07104",
        "source_url": "https://doi.org/10.1145/3431731",
        "source_kind": "manual_local_pdf",
        "section": "Secondary Priority",
        "paper_id": "MM9",
        "extra": "[DOI](https://doi.org/10.1145/3431731)",
        "links": ["https://doi.org/10.1145/3431731"],
    },
    {
        "filename": "date17.GATSim.pdf",
        "title": "GATSim: Abstract Timing Simulation of GPUs",
        "authors": "K. Punniyamurthy, B. Boroujerdian, A. Gerstlauer",
        "year": "2017",
        "venue": "DATE",
        "arxiv_id": None,
        "source_url": "https://slam.ece.utexas.edu/pubs/date17.GATSim.pdf",
        "source_kind": "manual_local_pdf",
        "section": "Highest Priority",
        "paper_id": "MM4",
        "extra": "[PDF](https://slam.ece.utexas.edu/pubs/date17.GATSim.pdf)",
        "links": ["https://slam.ece.utexas.edu/pubs/date17.GATSim.pdf"],
    },
    {
        "filename": "gong2017.pdf",
        "title": "Multi2Sim Kepler: A detailed architectural GPU simulator",
        "authors": "X. Gong, R. Ubal, D. Kaeli",
        "year": "2017",
        "venue": "ISPASS",
        "arxiv_id": None,
        "source_url": "https://doi.org/10.1109/ISPASS.2017.7975279",
        "source_kind": "manual_local_pdf",
        "section": "Additional Recovery",
        "paper_id": "MM12",
        "extra": "[DOI](https://doi.org/10.1109/ISPASS.2017.7975279)",
        "links": ["https://doi.org/10.1109/ISPASS.2017.7975279"],
    },
    {
        "filename": "gera_ispass18.pdf",
        "title": "Performance Characterisation and Simulation of Intel's Integrated GPU Architecture",
        "authors": "P. Gera, H. Kim, H. Kim, S. Hong, V. George, C.-K. Luk",
        "year": "2018",
        "venue": "ISPASS",
        "arxiv_id": None,
        "source_url": "https://ieeexplore.ieee.org/document/8366948",
        "source_kind": "manual_local_pdf",
        "section": "Highest Priority",
        "paper_id": "MM5",
        "extra": "[IEEE Xplore](https://ieeexplore.ieee.org/document/8366948)",
        "links": ["https://ieeexplore.ieee.org/document/8366948"],
    },
    {
        "filename": "kothapalli-performance-2009.pdf",
        "title": "A Performance Prediction Model for the CUDA GPGPU Platform",
        "authors": "K. Kothapalli, R. Mukherjee, M. S. Rehman, S. Patidar, P. J. Narayanan, K. Srinathan",
        "year": "2009",
        "venue": "HiPC",
        "arxiv_id": None,
        "source_url": "https://ieeexplore.ieee.org/document/5433179",
        "source_kind": "manual_local_pdf",
        "section": "Highest Priority",
        "paper_id": "MM7",
        "extra": "[IEEE Xplore](https://ieeexplore.ieee.org/document/5433179)",
        "links": ["https://ieeexplore.ieee.org/document/5433179"],
    },
    {
        "filename": "lee2013.pdf",
        "title": "Parallel GPU Architecture Simulation Framework Exploiting Work Allocation Unit Parallelism",
        "authors": "S. Lee, W. W. Ro",
        "year": "2013",
        "venue": "ISPASS",
        "arxiv_id": None,
        "source_url": "https://doi.org/10.1109/ISPASS.2013.6557151",
        "source_kind": "manual_local_pdf",
        "section": "Highest Priority",
        "paper_id": "MM2",
        "extra": "[DOI](https://doi.org/10.1109/ISPASS.2013.6557151)",
        "links": ["https://doi.org/10.1109/ISPASS.2013.6557151"],
    },
    {
        "filename": "malhotra2014.pdf",
        "title": "GpuTejas: A Parallel Simulator for GPU Architectures",
        "authors": "G. Malhotra, S. Goel, S. R. Sarangi",
        "year": "2014",
        "venue": "HiPC",
        "arxiv_id": None,
        "source_url": "https://www.cse.iitd.ac.in/~srsarangi/files/papers/gputejas.pdf",
        "source_kind": "manual_local_pdf",
        "section": "Highest Priority",
        "paper_id": "MM3",
        "extra": "[PDF](https://www.cse.iitd.ac.in/~srsarangi/files/papers/gputejas.pdf)",
        "links": ["https://www.cse.iitd.ac.in/~srsarangi/files/papers/gputejas.pdf"],
    },
    {
        "filename": "ubal2012.pdf",
        "title": "Multi2Sim: A Simulation Framework for CPU-GPU Computing",
        "authors": "R. Ubal, B. Jang, D. Schaa, P. Mistry, D. Kaeli",
        "year": "2012",
        "venue": "PACT",
        "arxiv_id": None,
        "source_url": "https://doi.org/10.1145/2370816.2370865",
        "source_kind": "manual_local_pdf",
        "section": "Highest Priority",
        "paper_id": "MM1",
        "extra": "[DOI](https://doi.org/10.1145/2370816.2370865)",
        "links": ["https://doi.org/10.1145/2370816.2370865"],
    },
    {
        "filename": "wang2019.pdf",
        "title": "A Hybrid Framework for Fast and Accurate GPU Performance Estimation through Source-Level Analysis and Trace-Based Simulation",
        "authors": "X. Wang, K. Huang, A. Knoll, X. Qian",
        "year": "2019",
        "venue": "HPCA",
        "arxiv_id": None,
        "source_url": "https://doi.org/10.1109/HPCA.2019.00062",
        "source_kind": "manual_local_pdf",
        "section": "Highest Priority",
        "paper_id": "MM8",
        "extra": "[DOI](https://doi.org/10.1109/HPCA.2019.00062)",
        "links": ["https://doi.org/10.1109/HPCA.2019.00062"],
    },
    {
        "filename": "ClusterSim_Modeling_Thread_Block_Clusters_in_Hopper_GPUs.pdf",
        "title": "ClusterSim: Modeling Thread Block Clusters in Hopper GPUs",
        "authors": "T. Lühnen, J. Behera, D. Tripathy, S. Lal",
        "year": "2025",
        "venue": "IISWC",
        "arxiv_id": None,
        "source_url": "https://doi.org/10.1109/IISWC63585.2025.00048",
        "source_kind": "manual_local_pdf",
        "section": "Highest Priority",
        "paper_id": "MM13",
        "extra": "[DOI](https://doi.org/10.1109/IISWC63585.2025.00048)",
        "links": ["https://doi.org/10.1109/IISWC63585.2025.00048"],
    },
]


def load_manifest() -> list[dict]:
    return json.loads(MANIFEST_PATH.read_text())


def write_manifest(records: list[dict]) -> None:
    MANIFEST_PATH.write_text(json.dumps(records, indent=2, ensure_ascii=True) + "\n")


def update_report(records: list[dict]) -> None:
    available = sum(1 for record in records if record.get("local_pdf") and Path(record["local_pdf"]).exists())
    resolved_sources = sum(1 for record in records if record.get("source_url") and not record.get("local_pdf"))
    unresolved = [
        record
        for record in records
        if not record.get("source_url") and not (record.get("local_pdf") and Path(record["local_pdf"]).exists())
    ]

    lines = [
        "# Paper Knowledge Base Report",
        "",
        f"- Total canonical papers: {len(records)}",
        f"- Available local PDFs: {available}",
        f"- Resolved remote PDF sources: {resolved_sources}",
        f"- Unresolved papers: {len(unresolved)}",
        "",
        "## Unresolved",
        "",
    ]

    if unresolved:
        for record in unresolved:
            lines.append(f"- {record['title']}")
    else:
        lines.append("- None")

    REPORT_PATH.write_text("\n".join(lines) + "\n")


def build_record(spec: dict, local_pdf: Path) -> dict:
    canonical_id = f"{spec['year']}__{slugify(spec['title'])}"
    return {
        "canonical_id": canonical_id,
        "title": spec["title"],
        "authors": spec["authors"],
        "year": spec["year"],
        "venue": spec["venue"],
        "arxiv_id": spec["arxiv_id"],
        "source_status": "downloaded",
        "source_url": spec["source_url"],
        "source_kind": spec["source_kind"],
        "local_pdf": str(local_pdf.resolve()),
        "occurrences": [
            {
                "source_list": "missing_milestones.md",
                "section": spec["section"],
                "subsection": "",
                "paper_id": spec["paper_id"],
                "title": spec["title"],
                "authors": spec["authors"],
                "year": spec["year"],
                "venue": spec["venue"],
                "extra": spec["extra"],
                "links": spec["links"],
                "arxiv_id": spec["arxiv_id"],
            }
        ],
    }


def main() -> None:
    records = load_manifest()
    by_title = {record["title"].lower(): record for record in records}
    by_id = {record["canonical_id"]: record for record in records}

    added: list[str] = []
    skipped: list[str] = []

    for spec in RECOVERED_PAPERS:
        source = UNRESOLVED_DIR / spec["filename"]
        if not source.exists():
            continue

        canonical_id = f"{spec['year']}__{slugify(spec['title'])}"
        if spec["title"].lower() in by_title or canonical_id in by_id:
            skipped.append(f"already present: {spec['title']}")
            continue

        destination = PDF_DIR / f"{canonical_id}.pdf"
        if destination.exists():
            skipped.append(f"destination exists without manifest entry: {destination.name}")
            continue

        shutil.move(str(source), str(destination))
        record = build_record(spec, destination)
        records.append(record)
        by_title[record["title"].lower()] = record
        by_id[record["canonical_id"]] = record
        added.append(record["canonical_id"])

    records.sort(key=lambda record: (record["title"].lower(), record["year"], record["canonical_id"]))
    write_manifest(records)
    update_report(records)

    unresolved_left = sorted(path.name for path in UNRESOLVED_DIR.glob("*.pdf"))

    print(f"[ingest] added={len(added)} skipped={len(skipped)} unresolved_left={len(unresolved_left)}")
    if added:
        print("[added]")
        for item in added:
            print(item)
    if skipped:
        print("[skipped]")
        for item in skipped:
            print(item)
    if unresolved_left:
        print("[remaining]")
        for item in unresolved_left:
            print(item)


if __name__ == "__main__":
    main()
