"""Verify every manifest-listed bundle byte, without dependencies or GPU use."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


def verify(root):
    root = Path(root).resolve(strict=True)
    rows = (root / "SHA256SUMS").read_text().splitlines()
    checked = total = 0
    seen = set()
    for row in rows:
        expected, relative = row.split("  ", 1)
        path = (root / relative).resolve(strict=True)
        if relative in seen or not path.is_relative_to(root) or not path.is_file():
            raise ValueError("Unsafe, duplicate or missing manifest path: " + relative)
        seen.add(relative)
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(block)
                total += len(block)
        if digest.hexdigest() != expected:
            raise ValueError("Hash mismatch: " + relative)
        checked += 1
    if not checked:
        raise ValueError("Empty package manifest")
    return checked, total


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    count, size = verify(args.package_root)
    print(f"PASS: {count} files, {size:,} bytes. No model loaded or GPU used.")


if __name__ == "__main__":
    main()
