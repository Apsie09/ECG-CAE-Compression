"""Download PTB-XL v1.0.3 and extract only the 100 Hz records and the metadata CSVs.

The official zip (~1.8 GB) also contains the 500 Hz records; those are skipped during
extraction. Pass --keep-zip to keep the archive, --with-500 to also extract records500/.
"""

from __future__ import annotations

import argparse
import urllib.request
import zipfile
from pathlib import Path

from tqdm import tqdm

from src.config import resolve

URL = ("https://physionet.org/static/published-projects/ptb-xl/"
       "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3.zip")


def download(url: str, dest: Path) -> None:
    if dest.exists():
        print(f"Using existing {dest}")
        return
    tmp = dest.with_suffix(".part")
    with urllib.request.urlopen(url) as r, open(tmp, "wb") as f:
        total = int(r.headers.get("Content-Length", 0))
        with tqdm(total=total, unit="B", unit_scale=True, desc="PTB-XL") as bar:
            while chunk := r.read(1 << 20):
                f.write(chunk)
                bar.update(len(chunk))
    tmp.rename(dest)


def extract(zip_path: Path, out_dir: Path, with_500: bool) -> None:
    wanted_dirs = ("records100/", "records500/") if with_500 else ("records100/",)
    with zipfile.ZipFile(zip_path) as zf:
        members = []
        for info in zf.infolist():
            # strip the top-level folder of the archive
            rel = info.filename.split("/", 1)[1] if "/" in info.filename else ""
            if not rel or info.is_dir():
                continue
            if (rel.endswith(".csv") and "/" not in rel) or rel.startswith(wanted_dirs):
                members.append((info, rel))
        for info, rel in tqdm(members, desc="extract"):
            target = out_dir / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info) as src, open(target, "wb") as dst:
                dst.write(src.read())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="data/ptbxl")
    ap.add_argument("--keep-zip", action="store_true")
    ap.add_argument("--with-500", action="store_true")
    args = ap.parse_args()

    out = resolve(args.out)
    if (out / "ptbxl_database.csv").exists() and (out / "records100").exists():
        print(f"PTB-XL already present in {out}")
        return
    out.mkdir(parents=True, exist_ok=True)
    zip_path = out.parent / "ptb-xl-1.0.3.zip"
    download(URL, zip_path)
    extract(zip_path, out, args.with_500)
    if not args.keep_zip:
        zip_path.unlink()
    print(f"Done: {out}")


if __name__ == "__main__":
    main()
