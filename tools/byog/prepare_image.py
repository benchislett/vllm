# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bake the pinned GSM8K data and build identity into the BYOG overlay."""

import hashlib
import json
import os
import sys
import urllib.request
from pathlib import Path

REVISION = "3101c7d5072418e28b9008a6636bde82a006892c"
HASHES = {
    "train": "17f347dc51477c50d4efb83959dbb7c56297aba886e5544ee2aaed3024813465",
    "test": "3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14",
}


def main() -> None:
    destination = Path(sys.argv[1])
    for split, expected in HASHES.items():
        url = (
            "https://raw.githubusercontent.com/openai/grade-school-math/"
            f"{REVISION}/grade_school_math/data/{split}.jsonl"
        )
        with urllib.request.urlopen(url, timeout=60) as response:
            content = response.read()
        if hashlib.sha256(content).hexdigest() != expected:
            raise ValueError(f"Unexpected {split} dataset content")
        (destination / "data" / f"{split}.jsonl").write_bytes(content)
    info = {
        "source_repository": os.environ["BYOG_SOURCE_REPOSITORY"],
        "source_commit": os.environ["BYOG_REQUESTED_COMMIT"],
        "harness_commit": os.environ["BYOG_REQUESTED_COMMIT"],
        "base_digest": sys.argv[2],
        "build_url": os.environ["BUILDKITE_BUILD_URL"],
        "dataset_revision": REVISION,
        "dataset_sha256": HASHES,
        "retention": "manual",
        "architecture": os.environ.get("BYOG_ARCH", "amd64"),
    }
    (destination / "build-info.json").write_text(json.dumps(info, indent=2) + "\n")


if __name__ == "__main__":
    main()
