"""Download a limited set of Argoverse 2 LiDAR logs using s5cmd."""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, List, Optional

BASE_URL = "https://s3.amazonaws.com/argoverse"
S3_URI_ROOT = "s3://argoverse"
LIDAR_PREFIX = "datasets/av2/lidar/"
XML_NS = "{http://s3.amazonaws.com/doc/2006-03-01/}"
DEFAULT_DEST = Path("..") / "argverse_data_preview"


@dataclass
class LogSelection:
    split: str
    log_ids: List[str]


class S3PrefixLister:
    def __init__(self, max_keys: int = 1000) -> None:
        self.max_keys = max_keys

    def _fetch(self, prefix: str, delimiter: str, continuation: Optional[str]) -> ET.Element:
        params = {
            "list-type": "2",
            "prefix": prefix,
            "delimiter": delimiter,
            "max-keys": str(self.max_keys),
        }
        if continuation:
            params["continuation-token"] = continuation
        url = f"{BASE_URL}?{urllib.parse.urlencode(params)}"
        with urllib.request.urlopen(url) as response:  # nosec B310
            data = response.read()
        return ET.fromstring(data)

    def iter_common_prefixes(self, prefix: str) -> Iterator[str]:
        continuation: Optional[str] = None
        while True:
            root = self._fetch(prefix, delimiter="/", continuation=continuation)
            for element in root.findall(f"{XML_NS}CommonPrefixes"):
                prefix_elem = element.find(f"{XML_NS}Prefix")
                if prefix_elem is None or prefix_elem.text is None:
                    continue
                yield prefix_elem.text
            token_elem = root.find(f"{XML_NS}NextContinuationToken")
            if token_elem is None or not token_elem.text:
                break
            continuation = token_elem.text


def collect_logs(split: str, *, limit: int, lister: S3PrefixLister) -> List[str]:
    if limit <= 0:
        return []
    prefix = f"{LIDAR_PREFIX}{split}/"
    found: List[str] = []
    for common in lister.iter_common_prefixes(prefix):
        if not common.startswith(prefix):
            continue
        suffix = common[len(prefix):].strip("/")
        if not suffix:
            continue
        found.append(suffix)
        if len(found) >= limit:
            break
    return found


def build_command(*, s5cmd: Path, source: str, dest: Path, concurrency: Optional[int],
                  part_size: Optional[int], extra_args: Iterable[str]) -> List[str]:
    command = [str(s5cmd), "--no-sign-request", "sync"]
    if concurrency:
        command += ["--concurrency", str(concurrency)]
    if part_size:
        command += ["--part-size", str(part_size)]
    command += list(extra_args)
    command += [source, str(dest)]
    return command


def run_command(command: List[str]) -> int:
    print("  Launching:")
    print("   " + " ".join(command))
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    assert process.stdout is not None
    try:
        for line in process.stdout:
            print("    " + line.rstrip())
    except KeyboardInterrupt:
        print("\nKeyboard interrupt detected; terminating s5cmd...", file=sys.stderr)
        process.terminate()
        raise
    return process.wait()


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download a small sample of Argoverse 2 LiDAR logs via s5cmd.")
    parser.add_argument("--dest", type=Path, default=DEFAULT_DEST,
                        help=f"Destination directory for the sample logs (default: {DEFAULT_DEST})")
    parser.add_argument("--split", action="append", choices=["train", "val", "test"],
                        help="Dataset split(s) to download from. Defaults to just 'val'.")
    parser.add_argument("--count", type=int, default=100,
                        help="Number of logs to download per selected split (default: %(default)s).")
    parser.add_argument("--s5cmd", type=Path,
                        default=Path(__file__).resolve().parent / "s5cmd" / "s5cmd.exe",
                        help="Path to the s5cmd executable (default: bundled binary).")
    parser.add_argument("--concurrency", type=int, default=16,
                        help="Concurrency level passed to s5cmd (default: %(default)s).")
    parser.add_argument("--part-size", type=int, default=128,
                        help="Multipart chunk size in MiB for s5cmd (default: %(default)s).")
    parser.add_argument("--extra-arg", action="append", default=[],
                        help="Repeat to add raw extra flags passed to s5cmd before the source/dest arguments.")
    parser.add_argument("--max-keys", type=int, default=1000,
                        help="How many prefixes to request per S3 listing page (default: %(default)s).")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip logs that already exist locally instead of re-syncing them.")
    return parser.parse_args(argv)


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = parse_args(argv)
    splits = args.split or ["val"]
    dest_root: Path = args.dest.resolve()
    dest_root.mkdir(parents=True, exist_ok=True)

    s5cmd_path: Path = args.s5cmd
    if not s5cmd_path.exists():
        raise FileNotFoundError(f"s5cmd executable not found: {s5cmd_path}")

    lister = S3PrefixLister(max_keys=args.max_keys)
    selections: List[LogSelection] = []
    for split in splits:
        print(f"Listing logs for split '{split}' (requesting up to {args.count})...")
        logs = collect_logs(split, limit=args.count, lister=lister)
        if not logs:
            print(f"  No logs discovered for split '{split}'.")
            continue
        print(f"  Found {len(logs)} logs for split '{split}'.")
        selections.append(LogSelection(split=split, log_ids=logs))

    total_logs = sum(len(sel.log_ids) for sel in selections)
    if total_logs == 0:
        print("No logs to download. Exiting.")
        return 0

    completed = 0
    start_time = time.monotonic()
    for sel in selections:
        split_dest = dest_root / sel.split
        split_dest.mkdir(parents=True, exist_ok=True)
        for log_id in sel.log_ids:
            completed += 1
            log_dest = split_dest / log_id
            if args.skip_existing and log_dest.exists():
                print(f"[{completed}/{total_logs}] Skipping existing log {sel.split}/{log_id}")
                continue
            print(f"[{completed}/{total_logs}] Downloading {sel.split}/{log_id}...")
            source = f"{S3_URI_ROOT}/{LIDAR_PREFIX}{sel.split}/{log_id}/*"
            command = build_command(
                s5cmd=s5cmd_path,
                source=source,
                dest=log_dest,
                concurrency=args.concurrency if args.concurrency > 0 else None,
                part_size=args.part_size if args.part_size > 0 else None,
                extra_args=args.extra_arg,
            )
            log_dest.mkdir(parents=True, exist_ok=True)
            result = run_command(command)
            if result != 0:
                print(f"s5cmd exited with code {result} while downloading {sel.split}/{log_id}", file=sys.stderr)
                return result
    elapsed = time.monotonic() - start_time
    print(f"Finished downloading {total_logs} logs in {elapsed/60:.2f} minutes. Data stored in {dest_root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
