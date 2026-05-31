#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import sys
import traceback
from pathlib import Path


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    failures = []
    total = 0
    for arg in sys.argv[1:]:
        path = Path(arg)
        module = load_module(path)
        for name in sorted(dir(module)):
            if not name.startswith("test_"):
                continue
            func = getattr(module, name)
            if not callable(func):
                continue
            total += 1
            try:
                func()
                print(f"PASS {path}:{name}")
            except Exception:
                failures.append((path, name, traceback.format_exc()))
                print(f"FAIL {path}:{name}")
    print(f"fallback pytest summary: {total - len(failures)} passed, {len(failures)} failed, {total} total")
    for path, name, tb in failures:
        print(f"\n--- failure: {path}:{name} ---")
        print(tb)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
