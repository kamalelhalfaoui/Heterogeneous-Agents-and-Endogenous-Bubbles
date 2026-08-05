"""
Regenerate every figure and table in the repository.

Usage
-----
    python scripts/run_all.py            # everything
    python scripts/run_all.py 8 9 13     # only those numbered scripts
    python scripts/run_all.py --list     # show what is available

The heavy parameter sweeps (scripts 8, 9, 11 and 13) take a few minutes each.
Everything is deterministic: the stochastic calibrations are seeded, so a clean
run reproduces the committed figures byte for byte.
"""

from __future__ import annotations

import runpy
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def discover() -> list[Path]:
    return sorted(
        p for p in HERE.glob("[0-9][0-9]_*.py") if p.name != Path(__file__).name
    )


def main(argv: list[str]) -> int:
    scripts = discover()

    if "--list" in argv:
        for s in scripts:
            print(f"  {s.name}")
        return 0

    wanted = [a for a in argv if a.isdigit()]
    if wanted:
        keys = {f"{int(w):02d}" for w in wanted}
        scripts = [s for s in scripts if s.name[:2] in keys]
        if not scripts:
            print("No scripts matched.", file=sys.stderr)
            return 1

    total = time.perf_counter()
    failures: list[tuple[str, Exception]] = []

    for script in scripts:
        print(f"\n[{script.name}]")
        t0 = time.perf_counter()
        try:
            runpy.run_path(str(script), run_name="__main__")
        except Exception as exc:  # keep going; report at the end
            failures.append((script.name, exc))
            print(f"  FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        else:
            print(f"  done in {time.perf_counter() - t0:.1f}s")

    elapsed = time.perf_counter() - total
    print(f"\n{len(scripts) - len(failures)}/{len(scripts)} scripts succeeded "
          f"in {elapsed:.1f}s")
    if failures:
        print("\nFailures:", file=sys.stderr)
        for name, exc in failures:
            print(f"  {name}: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
