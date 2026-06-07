#!/usr/bin/env python3
"""Verify that each TensorRT layer report contains INT8 tensor formats."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", nargs="+", type=Path)
    args = parser.parse_args()

    failed = False
    for report in args.reports:
        if not report.is_file():
            print(f"FAIL {report}: missing layer report")
            failed = True
            continue
        text = report.read_text(errors="replace")
        counts = {
            precision: len(re.findall(rf"\b{precision}\b", text, flags=re.IGNORECASE))
            for precision in ("int8", "half", "float")
        }
        if counts["int8"] == 0:
            print(f"FAIL {report}: no INT8 tensor formats found")
            failed = True
        else:
            print(
                f"PASS {report}: INT8={counts['int8']} "
                f"FP16={counts['half']} FP32={counts['float']}"
            )

    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
