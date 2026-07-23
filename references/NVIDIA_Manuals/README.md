# NVIDIA Manuals Mirror & Crawler

Local Markdown mirror of NVIDIA's online documentation, refreshed on
demand from `docs.nvidia.com`. Each subfolder is one product manual.

## Active doc folders

| Folder | Crawler | Source URL |
| --- | --- | --- |
| `ampere_compatibility/`    | single | https://docs.nvidia.com/cuda/ampere-compatibility-guide/ |
| `ampere_tuning/`           | single | https://docs.nvidia.com/cuda/ampere-tuning-guide/ |
| `best_practices/`          | single | https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/ |
| `blackwell_compatibility/` | single | https://docs.nvidia.com/cuda/blackwell-compatibility-guide/ |
| `blackwell_tuning/`        | single | https://docs.nvidia.com/cuda/blackwell-tuning-guide/ |
| `cuda_binary_utility/`     | single | https://docs.nvidia.com/cuda/cuda-binary-utilities/ |
| `cuda_guide/`              | multi  | https://docs.nvidia.com/cuda/cuda-programming-guide/ |
| `cuda_inline_ptx/`         | single | https://docs.nvidia.com/cuda/inline-ptx-assembly/ |
| `cuda_tile_ir/`            | multi  | https://docs.nvidia.com/cuda/tile-ir/latest/ |
| `cutile_python/`           | multi  | https://docs.nvidia.com/cuda/cutile-python/ |
| `cutlass_cute/`            | multi  | https://docs.nvidia.com/cutlass/latest/ |
| `nvcc_guide/`              | single | https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/ |
| `nvvm_ir/`                 | single | https://docs.nvidia.com/cuda/nvvm-ir-spec/ |
| `ptx_compier_api/`         | single | https://docs.nvidia.com/cuda/ptx-compiler-api/ |
| `ptx_isa/`                 | single | https://docs.nvidia.com/cuda/parallel-thread-execution/ |
| `ptx_writer_guide/`        | single | https://docs.nvidia.com/cuda/ptx-writers-guide-to-interoperability/ |

The companion folder `solbench_problems/` holds 235 problem specs
scraped from `research.nvidia.com/benchmarks/sol-execbench/` and is
maintained by `scripts/sol_execbench_crawler.py`, not by the two doc
crawlers below.

## Crawler scripts (in `scripts/`)

- `crawler_singlefile.py` — Sphinx doc with a `contents.html` TOC
  (single-page or shallow multi-page).
- `crawler_multiTOC.py` — Sphinx doc spread across many HTML pages
  with sidebar / toctree navigation. BFS-discovers pages.
- `sol_execbench_crawler.py` — Playwright scraper for the SOL-ExecBench
  problem set (different domain, separate workflow; not covered below).

---

## How to refresh all 16 folders

Run from `references/NVIDIA_Manuals/`. The recipe assumes
[`uv`](https://docs.astral.sh/uv/) is on `PATH` (`/usr/bin/uv` in
this environment; install via `curl -LsSf https://astral.sh/uv/install.sh | sh`
if missing).

### 1. One-time environment setup (or after `requirements.txt` changes)

```bash
cd references/NVIDIA_Manuals
uv venv .venv                                    # uv-managed venv at .venv/
uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python -c "import requests, bs4, lxml, brotli; print('deps ok')"
```

`brotli` is **required**: `docs.nvidia.com` serves some pages with
`Content-Encoding: br`. Without the decompressor, `requests` returns
raw compressed bytes as "text" and the crawler writes binary mojibake.

### 2. Dry-run page discovery (multi-page targets only)

Always check that page counts are sensible before doing real crawls.
`crawler_multiTOC.py` caps page discovery at `--max-pages` (default
200). Set it to a value clearly above the expected count.

```bash
PY=.venv/bin/python
for url in \
    https://docs.nvidia.com/cuda/cuda-programming-guide/index.html \
    https://docs.nvidia.com/cuda/tile-ir/latest/index.html \
    https://docs.nvidia.com/cuda/cutile-python/index.html \
    https://docs.nvidia.com/cutlass/latest/index.html ; do
  echo "===== $url ====="
  $PY scripts/crawler_multiTOC.py --list-pages --max-pages 5000 "$url" \
    2>&1 | grep -E "Pages    :"
done
```

Typical counts (as of 2026-05-12): CUDA Programming Guide ≈48, Tile IR
≈13, cuTile Python ≈12, CUTLASS ≈81. All comfortably under 500.

### 3. Pre-flight manifest (state record, useful for diff later)

```bash
DATE=$(date +%Y-%m-%d)
{
  printf "# pre-refresh manifest  generated: %s\n" "$(date -Iseconds)"
  printf "# folder | n_md | bytes_md_only | source_url\n"
  for d in */; do
    name=${d%/}
    case "$name" in scripts|solbench_problems|*.bak-*|*.corrupt-*|.venv) continue ;; esac
    cnt=$(find "$d" -maxdepth 1 -name "*.md" 2>/dev/null | wc -l)
    sz=$(find "$d" -maxdepth 1 -name "*.md" -exec cat {} \; 2>/dev/null | wc -c)
    src=$(grep -m1 -E "^source:" "$d/00_index.md" 2>/dev/null | sed "s/source: //")
    printf "%-26s | %4d | %10d | %s\n" "$name" "$cnt" "$sz" "$src"
  done
} > "manifest_before_$DATE.txt"
```

### 4. Back up each folder (copy, don't move)

```bash
DATE=$(date +%Y-%m-%d)
for d in ampere_compatibility ampere_tuning best_practices \
         blackwell_compatibility blackwell_tuning cuda_binary_utility \
         cuda_guide cuda_inline_ptx cuda_tile_ir cutile_python \
         cutlass_cute nvcc_guide nvvm_ir ptx_compier_api \
         ptx_isa ptx_writer_guide; do
  [ -d "$d.bak-$DATE" ] || cp -a "$d" "$d.bak-$DATE"
done
```

### 5. Crawl every target into a clean staging dir `<folder>.new/`

```bash
PY=.venv/bin/python
LOG=crawl_log_$(date +%Y-%m-%d).txt
: > "$LOG"

crawl_single() {
  local folder=$1 url=$2
  echo "===== [SINGLE: $folder] $url =====" | tee -a "$LOG"
  rm -rf "$folder.new"
  $PY scripts/crawler_singlefile.py "$url" \
      -o "$folder.new" -d 1.5 --download-images >> "$LOG" 2>&1
}
crawl_multi() {
  local folder=$1 url=$2
  echo "===== [MULTI:  $folder] $url =====" | tee -a "$LOG"
  rm -rf "$folder.new"
  $PY scripts/crawler_multiTOC.py "$url" \
      -o "$folder.new" -d 1.5 --max-pages 500 --download-images >> "$LOG" 2>&1
}

# 12 single-page docs
crawl_single ampere_compatibility    https://docs.nvidia.com/cuda/ampere-compatibility-guide/
crawl_single ampere_tuning           https://docs.nvidia.com/cuda/ampere-tuning-guide/
crawl_single best_practices          https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/
crawl_single blackwell_compatibility https://docs.nvidia.com/cuda/blackwell-compatibility-guide/
crawl_single blackwell_tuning        https://docs.nvidia.com/cuda/blackwell-tuning-guide/
crawl_single cuda_binary_utility     https://docs.nvidia.com/cuda/cuda-binary-utilities/
crawl_single cuda_inline_ptx         https://docs.nvidia.com/cuda/inline-ptx-assembly/
crawl_single nvcc_guide              https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/
crawl_single nvvm_ir                 https://docs.nvidia.com/cuda/nvvm-ir-spec/
crawl_single ptx_compier_api         https://docs.nvidia.com/cuda/ptx-compiler-api/
crawl_single ptx_isa                 https://docs.nvidia.com/cuda/parallel-thread-execution/
crawl_single ptx_writer_guide        https://docs.nvidia.com/cuda/ptx-writers-guide-to-interoperability/

# 4 multi-page docs
crawl_multi cutile_python https://docs.nvidia.com/cuda/cutile-python/index.html
crawl_multi cuda_tile_ir  https://docs.nvidia.com/cuda/tile-ir/latest/index.html
crawl_multi cuda_guide    https://docs.nvidia.com/cuda/cuda-programming-guide/index.html
crawl_multi cutlass_cute  https://docs.nvidia.com/cutlass/latest/index.html
```

`-d 1.5` (1.5 s between requests) is the polite default for
`docs.nvidia.com`. `--download-images` is required to keep the
markdown renderable offline; without it `<img>` tags point at remote
URLs.

A full refresh of all 16 takes ~30-45 min depending on network
latency. The crawl is sequential by design — `docs.nvidia.com` is
fronted by a CDN that throttles concurrent clients.

### 6. Validate every staging dir before swapping

The patched crawlers refuse to write decoded-as-binary content (see
"Crawler safeguards" below). Even so, validate before swap:

```bash
for d in ampere_compatibility ampere_tuning best_practices \
         blackwell_compatibility blackwell_tuning cuda_binary_utility \
         cuda_guide cuda_inline_ptx cuda_tile_ir cutile_python \
         cutlass_cute nvcc_guide nvvm_ir ptx_compier_api \
         ptx_isa ptx_writer_guide; do
  total=$(find "$d.new" -maxdepth 1 -name "*.md" 2>/dev/null | wc -l)

  # Mojibake (U+FFFD density >5%)
  bad=0
  while IFS= read -r f; do
    sz=$(stat -c%s "$f"); [ "$sz" -lt 100 ] && continue
    rc=$(grep -ao $'\xef\xbf\xbd' "$f" 2>/dev/null | wc -l)
    awk -v r="$rc" -v s="$sz" 'BEGIN{exit !((3*r/s) > 0.05)}' && bad=$((bad+1))
  done < <(find "$d.new" -maxdepth 1 -name "*.md" 2>/dev/null)

  # Off-by-one / broken index links
  bidx=0; itot=0
  while read -r link; do
    fn=$(echo "$link" | sed -E 's|.*\]\(\./||; s|\).*||')
    itot=$((itot+1))
    [ -f "$d.new/$fn" ] || bidx=$((bidx+1))
  done < <(grep -oE '\]\(\./[^)]+\.md\)' "$d.new/00_index.md" 2>/dev/null)

  imgs=$(find "$d.new/images" -type f 2>/dev/null | wc -l)
  printf "%-26s %4d files | %3d corrupted | %3d/%3d broken_idx | %4d imgs\n" \
         "$d" "$total" "$bad" "$bidx" "$itot" "$imgs"
done
```

A clean run shows `0 corrupted` and `0/<itot> broken_idx` for every
target. If any non-zero appears, **do not swap** — investigate the
crawl log and re-run the affected target.

### 7. Atomic swap into live position

```bash
for d in ampere_compatibility ampere_tuning best_practices \
         blackwell_compatibility blackwell_tuning cuda_binary_utility \
         cuda_guide cuda_inline_ptx cuda_tile_ir cutile_python \
         cutlass_cute nvcc_guide nvvm_ir ptx_compier_api \
         ptx_isa ptx_writer_guide; do
  [ -f "$d.new/00_index.md" ] || { echo "SKIP $d (no index)"; continue; }
  rm -rf "$d" && mv "$d.new" "$d"
done
```

### 8. Post-flight manifest

```bash
DATE=$(date +%Y-%m-%d)
{
  printf "# post-refresh manifest  generated: %s\n" "$(date -Iseconds)"
  printf "# folder | n_md | bytes_md_only | n_images | source_url\n"
  for d in */; do
    name=${d%/}
    case "$name" in scripts|solbench_problems|*.bak-*|.venv) continue ;; esac
    cnt=$(find "$d" -maxdepth 1 -name "*.md" 2>/dev/null | wc -l)
    sz=$(find "$d" -maxdepth 1 -name "*.md" -exec cat {} \; 2>/dev/null | wc -c)
    imgn=$(find "$d/images" -type f 2>/dev/null | wc -l)
    src=$(grep -m1 -E "^source:" "$d/00_index.md" 2>/dev/null | sed "s/source: //")
    printf "%-26s | %4d | %10d | %4d | %s\n" "$name" "$cnt" "$sz" "$imgn" "$src"
  done
} > "manifest_after_$DATE.txt"
diff "manifest_before_$DATE.txt" "manifest_after_$DATE.txt" | head -40
```

### 9. Cleanup once satisfied

```bash
# After confirming the new state is good, remove backups
rm -rf *.bak-*

# Optionally remove the manifests and log (they're audit records)
# rm manifest_before_*.txt manifest_after_*.txt crawl_log_*.txt
```

Keep `.venv/` — it's the reusable environment for the next refresh.

---

## Crawler safeguards (patched 2026-05-12)

The previous unpatched crawlers produced silent binary corruption on
multi-page docs (~12-46% of files in cuda_guide, cuda_tile_ir,
cutile_python, cutlass_cute). The fix in both `crawler_singlefile.py`
and `crawler_multiTOC.py` covers three failure modes:

- **Encoding override removed.** The line
  `r.encoding = r.apparent_encoding or "utf-8"` mis-guessed
  brotli-compressed responses as Latin-1. The crawler now trusts the
  server's declared charset and defaults to UTF-8 when `Content-Type`
  omits it.
- **Decode sanity check.** Each fetched page is rejected with a logged
  error if (a) its first 4 KB contains neither `<html` nor `<!doctype`,
  or (b) it contains more than 16 U+FFFD replacement chars at >5%
  density. Both signal a decode failure that previously slipped through.
- **Index off-by-one.** `_save_index` previously used
  `enumerate(..., 0)` while `_write_split` used `enumerate(..., 1)`.
  Result: 00_index.md linked to `./0000_*.md` but the actual file was
  `0001_*.md`. Both now use `enumerate(..., 1)`.

## Reproducibility

- `requirements.txt` pins lower bounds (`>=`) for the four direct deps;
  uv resolves the full set deterministically against PyPI at install
  time. To freeze a known-good full set after a successful refresh:
  `uv pip freeze --python .venv/bin/python > requirements.lock.txt`.
- Crawl scripts log every fetch to `crawl_log_YYYY-MM-DD.txt`. The
  log + manifest pair is enough to audit what changed.
- The pinned URLs above use `latest/` for CUTLASS and Tile IR; both
  float. To pin a specific version, replace `latest/` with e.g.
  `cutlass/4.5.0/` or `cuda/tile-ir/13.2/`. Check the upstream archive
  URL still resolves before using it.
