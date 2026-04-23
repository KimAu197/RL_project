"""Download and unpack the Spider dataset.

Spider is distributed by the Yale LILY group as a single zip hosted on
Google Drive. The current release is ``spider_data.zip``
(file id ``1403EGqzIDoHMdQF4c9Bkyl7dZLZ5Wt6J``). The download flow has two
paths:

1. If ``gdown`` is installed, we use it (it handles the Drive confirm-token
   dance correctly). This is the recommended path.
2. Otherwise we fall back to a plain HTTPS fetch from a direct URL. This
   works when Drive doesn't inject a confirm page, and gives the user a
   clear manual-fallback message if it fails.

The zip extracts into a top-level ``spider_data/`` directory. To keep the
rest of the repo (configs, scripts, dataset loaders) working with the
historical ``datasets/spider/`` path, we rename that directory to
``spider/`` after extraction.

Final on-disk layout:

    <dest>/spider/
        database/<db_id>/<db_id>.sqlite
        tables.json
        train_spider.json
        train_others.json
        dev.json

If you hit rate limits or the Drive link rots, simply:

    pip install gdown
    gdown --id 1403EGqzIDoHMdQF4c9Bkyl7dZLZ5Wt6J -O datasets/spider.zip

or download manually from https://yale-lily.github.io/spider and place
``spider.zip`` at ``datasets/spider.zip`` before re-running.
"""
from __future__ import annotations

import argparse
import shutil
import sys
import zipfile
from pathlib import Path

DRIVE_FILE_ID = "1403EGqzIDoHMdQF4c9Bkyl7dZLZ5Wt6J"
DIRECT_URL = f"https://drive.usercontent.google.com/download?id={DRIVE_FILE_ID}&export=download&confirm=t"
PROJECT_PAGE = "https://yale-lily.github.io/spider"
ARCHIVE_ROOT_NAME = "spider_data"
TARGET_ROOT_NAME = "spider"


def _download_with_gdown(dest: Path) -> bool:
    try:
        import gdown  # type: ignore
    except ImportError:
        return False
    print("[download_spider] using gdown")
    url = f"https://drive.google.com/uc?id={DRIVE_FILE_ID}"
    try:
        gdown.download(url, str(dest), quiet=False)
    except Exception as exc:
        print(f"[download_spider] gdown failed: {exc}", file=sys.stderr)
        return False
    return dest.exists() and dest.stat().st_size > 0


def _download_direct(url: str, dest: Path) -> bool:
    import requests
    print(f"[download_spider] downloading directly from {url}")
    try:
        with requests.get(url, stream=True, timeout=120) as r:
            r.raise_for_status()
            total = int(r.headers.get("content-length", 0))
            written = 0
            with open(dest, "wb") as f:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    if not chunk:
                        continue
                    f.write(chunk)
                    written += len(chunk)
                    if total:
                        pct = 100.0 * written / total
                        print(f"\r[download_spider] {written/1e6:.1f}MB / {total/1e6:.1f}MB ({pct:.1f}%)", end="")
            if total:
                print()
    except Exception as exc:
        print(f"[download_spider] direct download failed: {exc}", file=sys.stderr)
        return False
    return dest.exists() and dest.stat().st_size > 0


def _unzip(zip_path: Path, dest: Path) -> None:
    print(f"[download_spider] extracting {zip_path} -> {dest}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest)


def _normalize_extracted_root(dest_root: Path) -> Path:
    """Ensure the extracted tree lives at ``dest_root / TARGET_ROOT_NAME``.

    The current Spider release unzips into ``spider_data/``. Older releases
    unzipped into ``spider/``. We normalize to ``spider/`` so downstream
    configs and scripts don't need to change.
    """
    target = dest_root / TARGET_ROOT_NAME
    archive = dest_root / ARCHIVE_ROOT_NAME
    if target.exists():
        return target
    if archive.exists():
        print(f"[download_spider] renaming {archive} -> {target}")
        archive.rename(target)
        return target
    return target


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download Spider dataset")
    parser.add_argument("--dest", type=Path, default=Path("datasets"), help="destination root")
    parser.add_argument("--url", type=str, default=DIRECT_URL, help="direct download URL for spider.zip")
    parser.add_argument("--force", action="store_true", help="redownload even if cached")
    args = parser.parse_args(argv)

    dest_root: Path = args.dest
    dest_root.mkdir(parents=True, exist_ok=True)
    zip_path = dest_root / "spider.zip"
    extracted_root = dest_root / TARGET_ROOT_NAME
    archive_root = dest_root / ARCHIVE_ROOT_NAME

    if args.force and zip_path.exists():
        zip_path.unlink()
    if args.force:
        for p in (extracted_root, archive_root):
            if p.exists():
                shutil.rmtree(p)

    if not (zip_path.exists() and zip_path.stat().st_size > 0):
        ok = _download_with_gdown(zip_path) or _download_direct(args.url, zip_path)
        if not ok:
            print(
                "[download_spider] could not download spider.zip automatically.\n"
                "Manual options:\n"
                f"  1) pip install gdown && gdown --id {DRIVE_FILE_ID} -O {zip_path}\n"
                f"  2) visit {PROJECT_PAGE} and place spider.zip at {zip_path}",
                file=sys.stderr,
            )
            return 1
    else:
        print(f"[download_spider] {zip_path} already present, skipping download")

    if not extracted_root.exists() and not archive_root.exists():
        _unzip(zip_path, dest_root)

    extracted_root = _normalize_extracted_root(dest_root)

    expected = [
        extracted_root / "tables.json",
        extracted_root / "train_spider.json",
        extracted_root / "dev.json",
    ]
    missing = [str(p) for p in expected if not p.exists()]
    if missing:
        print(f"[download_spider] extraction did not produce expected files: {missing}", file=sys.stderr)
        return 2

    print(f"[download_spider] Spider is ready at {extracted_root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
