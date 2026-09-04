#!/usr/bin/env python3
"""Hand retouch: eye + mouth legibility on the ramp-80 logo grid.

The converter renders anatomy honestly, but a ~5-char eye and a 1-char
mouth slit fall under its legibility floor — they blur into the skull
mass. This patch sculpts the head features with sparse dither glyphs
(░ rim, space core) on top of fresh converter output, so regenerating
the logo reapplies the sculpt deterministically.

Input : docs/brand/ascii-logo-ramp-80.txt
        (fresh: python3 tools/ascii_convert.py --width 80 --mode ramp \\
                 --out docs/brand/ascii-logo-ramp-80.txt)
Output: docs/brand/ascii-logo-ramp-80-retouched.txt

Every edit is (line, col, expected, replacement): `expected` is asserted
against the base grid, so converter drift fails loudly here instead of
silently sculpting the wrong pixel. Coordinates are pinned to the
ramp-80 grid (80x40, CELL_ASPECT law).
"""
from pathlib import Path

BASE = Path("docs/brand/ascii-logo-ramp-80.txt")
OUT = Path("docs/brand/ascii-logo-ramp-80-retouched.txt")

EDITS = [
    # --- eye: negative-space core + ░ rim (source: white slit in skull)
    (2, 38, "█", "░"),
    (2, 39, "▒", " "),
    (2, 40, "▒", " "),
    (2, 41, "▓", " "),
    (2, 42, "█", "░"),
    (3, 39, "█", "░"),
    (3, 40, "█", "▒"),
    (3, 41, "█", "░"),
    # --- mouth: the bite slash — sparse dither, widening wedge below
    (4, 41, "▒", "░"),
    (4, 42, "▒", " "),
    (4, 43, "▒", "░"),
    (4, 44, "▒", " "),
    (5, 40, "▒", "░"),
    (5, 41, "░", " "),
    (5, 42, "▒", " "),
    (5, 43, "▒", "░"),
    (6, 45, "▓", "░"),
    (6, 46, "▒", " "),
    (6, 47, "▒", "░"),
    (6, 48, "░", " "),
    (7, 40, "░", " "),
    (7, 42, "▒", " "),
    (7, 44, "▒", " "),
]


def main() -> None:
    grid = [list(line) for line in BASE.read_text().splitlines()]
    for line, col, expected, replacement in EDITS:
        actual = grid[line][col]
        if actual != expected:
            raise SystemExit(
                f"drift at line {line} col {col}: expected {expected!r}, "
                f"found {actual!r} — re-derive coordinates against fresh output"
            )
        grid[line][col] = replacement
    OUT.write_text("\n".join("".join(row) for row in grid) + "\n")
    print(f"{OUT}: {len(EDITS)} glyphs retouched")


if __name__ == "__main__":
    main()
