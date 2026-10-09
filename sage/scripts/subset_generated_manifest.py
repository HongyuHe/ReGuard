#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a bounded Sage generated-trace manifest.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, required=True)
    args = parser.parse_args()

    if args.count <= 0:
        raise ValueError("--count must be positive")
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    entries = list(payload.get("generated_entries", []))
    if len(entries) < args.count:
        raise RuntimeError(f"manifest has {len(entries)} entries, fewer than requested {args.count}")
    payload["generated_entries"] = entries[: args.count]
    payload["source_manifest"] = str(args.input)
    payload["selected_entry_count"] = int(args.count)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
