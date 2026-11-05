"""Download and extract Google Speech Commands v0.02 into data/speech_commands/.

Usage:  uv run python src/download_data.py          # skip if already present
        uv run python src/download_data.py --force  # re-download and re-extract

The archive (~2.3 GB) is fetched from Google's official URL, extracted
(~3.8 GB), checked, and then deleted unless --keep-archive is given.
Dataset license: CC BY 4.0 (Warden, 2018, arXiv:1804.03209).
"""

import argparse
import shutil
import sys
import tarfile
import time
import urllib.request
from pathlib import Path

URL = "http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DEST = DATA_DIR / "speech_commands"
ARCHIVE = DATA_DIR / "speech_commands_v0.02.tar.gz"
CACHE = DATA_DIR / "speech_cache"
EXPECTED_WAVS = 105_835  # 105,829 word clips + 6 background-noise files
N_WORDS = 35


def is_complete(dest: Path = DEST) -> bool:
    if not (dest / "validation_list.txt").exists() or not (dest / "testing_list.txt").exists():
        return False
    return sum(1 for _ in dest.rglob("*.wav")) == EXPECTED_WAVS


def download(url: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    t0 = time.time()
    with urllib.request.urlopen(url) as r, open(tmp, "wb") as f:
        total = int(r.headers.get("Content-Length", 0))
        done = 0
        while chunk := r.read(1 << 20):
            f.write(chunk)
            done += len(chunk)
            if total:
                speed = done / max(time.time() - t0, 1e-6) / 1e6
                print(f"\r  downloading {done / 1e9:.2f}/{total / 1e9:.2f} GB "
                      f"({done / total:.0%}, {speed:.0f} MB/s)", end="", flush=True)
    print()
    if total and done != total:
        tmp.unlink()
        raise RuntimeError(f"incomplete download: {done} of {total} bytes")
    tmp.rename(path)  # only a complete file gets the final name


def extract(archive: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    print(f"  extracting to {dest} ...", flush=True)
    with tarfile.open(archive, "r:gz") as tar:
        # filter="data" blocks absolute paths, '..' and links outside dest.
        tar.extractall(dest, filter="data")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--force", action="store_true", help="re-download and re-extract")
    ap.add_argument("--keep-archive", action="store_true", help="keep the .tar.gz after extracting")
    args = ap.parse_args()

    if is_complete() and not args.force:
        print(f"Dataset already present and complete at {DEST} (nothing to do).")
        return 0

    if args.force:
        shutil.rmtree(DEST, ignore_errors=True)
        shutil.rmtree(CACHE, ignore_errors=True)  # cached waveforms built from the old copy
        ARCHIVE.unlink(missing_ok=True)

    free = shutil.disk_usage(DATA_DIR.parent).free / 1e9
    if free < 7:
        print(f"Need ~7 GB free (archive + extracted), only {free:.1f} GB available.", file=sys.stderr)
        return 1

    if not ARCHIVE.exists():
        print(f"Downloading {URL}")
        download(URL, ARCHIVE)
    extract(ARCHIVE, DEST)

    if not is_complete():
        print(f"Extraction finished but {DEST} doesn't contain the expected "
              f"{EXPECTED_WAVS} .wav files. Re-run with --force.", file=sys.stderr)
        return 1
    words = [p for p in DEST.iterdir() if p.is_dir() and not p.name.startswith("_")]
    assert len(words) == N_WORDS, f"expected {N_WORDS} word folders, found {len(words)}"

    if not args.keep_archive:
        ARCHIVE.unlink()
    print(f"Done: {EXPECTED_WAVS} .wav files, {N_WORDS} words in {DEST}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
