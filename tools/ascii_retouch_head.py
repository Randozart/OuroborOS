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
    # --- eye: a tilted SLIT, not a hole — 3 connected dark runs
    #     (row1 col41 -> row2 cols39-40 = '/' lean, per the source
    #     parallelogram); ░ rim left, ▓ stays as the bright right edge.
    #     Row 3 untouched (v1's rim there read as a second smudge).
    (1, 41, "█", " "),
    (2, 38, "█", "░"),
    (2, 39, "▒", " "),
    (2, 40, "▒", " "),
    # --- mouth: ONE continuous descending slit — snout rows stay solid
    #     (v1 punched holes above the true slash). Core path:
    #     row4 col41 -> row5 cols41-42 -> row6 cols45-46 -> the open
    #     wedge / bite point. ░ shoulders keep it a line, not a crater.
    (4, 41, "▒", " "),
    (4, 42, "▒", "░"),
    (5, 40, "▒", "░"),
    (5, 41, "░", " "),
    (5, 42, "▒", " "),
    (5, 43, "▒", "░"),
    (6, 44, "▓", "░"),
    (6, 45, "▓", " "),
    (6, 46, "▒", " "),
    (6, 47, "▒", "░"),
    # row 7 wedge: converter already renders it open — untouched.
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
