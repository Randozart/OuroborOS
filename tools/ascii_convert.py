#!/usr/bin/env python3
"""Convert docs/brand/ouro-logo.svg to terminal ASCII art.

The roundness law: terminal cells are ~1 wide x 2 tall, so art only
looks round when CHARS_WIDE == 2 * LINES_TALL. CELL_ASPECT encodes the
cell shape; every output grid derives its height from its width through
it — no converter may map the logo into a square character grid again
(that is how the tall-oval ascii-logo-trimmed.txt happened).

Modes:
  solid  one pixel-sample per character cell -> '█' / space
  half   two pixel-rows per line -> '▀' / '▄' / '█' / space (crisper
         curves at the same physical size: W x 2H real pixels)

Usage:
  ascii_convert.py --width 120 --mode solid --out docs/brand/ascii-logo.txt
"""
import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

from PIL import Image

CELL_ASPECT = 0.5  # char width / char height on a classic terminal
SUPER = 4          # supersampling factor per cell
DEFAULT_THRESHOLD = 0.40  # cell coverage needed to ink a pixel


def rasterize(svg: Path, px: int) -> Image.Image:
    with tempfile.NamedTemporaryFile(suffix=".png") as tmp:
        subprocess.run(
            ["rsvg-convert", "-w", str(px), "-h", str(px), str(svg), "-o", tmp.name],
            check=True,
        )
        return Image.open(tmp.name).convert("RGBA")


def content_grid(svg: Path, width: int, pixel_rows: int, threshold: float) -> list[list[int]]:
    """Alpha coverage grid of width x pixel_rows cells, cropped to the
    logo's bounding box (which is square — the serpent is round)."""
    im = rasterize(svg, 1024)
    bbox = im.getbbox()
    im = im.crop(bbox)
    side = max(im.size)
    padded = Image.new("RGBA", (side, side), (0, 0, 0, 0))
    padded.paste(im, ((side - im.width) // 2, (side - im.height) // 2))

    ss_w, ss_h = width * SUPER, pixel_rows * SUPER
    small = padded.resize((ss_w, ss_h), Image.Resampling.LANCZOS)
    data = small.getchannel("A").tobytes()  # flat L-mode pixel bytes

    grid = [[0] * width for _ in range(pixel_rows)]
    for r in range(pixel_rows):
        for c in range(width):
            total = 0
            for y in range(r * SUPER, (r + 1) * SUPER):
                row_off = y * ss_w
                for x in range(c * SUPER, (c + 1) * SUPER):
                    total += data[row_off + x]
            grid[r][c] = 1 if total / (SUPER * SUPER) >= threshold else 0
    return grid


def emit_solid(grid: list[list[int]]) -> str:
    return "\n".join(
        "".join("█" if cell else " " for cell in row).rstrip() for row in grid
    )


HALF_GLYPH = {(1, 1): "█", (1, 0): "▀", (0, 1): "▄", (0, 0): " "}


def emit_half(grid: list[list[int]]) -> str:
    # grid has 2*lines pixel rows; pair them per output line
    lines = []
    for r in range(0, len(grid), 2):
        top, bot = grid[r], grid[r + 1]
        lines.append(
            "".join(HALF_GLYPH[(t, b)] for t, b in zip(top, bot)).rstrip()
        )
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--svg", type=Path, default=Path("docs/brand/ouro-logo.svg"))
    ap.add_argument("--width", type=int, required=True, help="chars wide")
    ap.add_argument("--mode", choices=["solid", "half"], required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    args = ap.parse_args()

    threshold = args.threshold

    lines = args.width // 2  # the roundness law: chars_wide == 2 * lines
    pixel_rows = lines if args.mode == "solid" else lines * 2
    grid = content_grid(args.svg, args.width, pixel_rows, threshold)

    text = (emit_solid if args.mode == "solid" else emit_half)(grid)
    # trim leading/trailing blank lines, keep interior blank lines
    rows = text.splitlines()
    while rows and not rows[0].strip():
        rows.pop(0)
    while rows and not rows[-1].strip():
        rows.pop()
    args.out.write_text("\n".join(rows) + "\n")
    print(f"{args.out}: {args.width} wide x {len(rows)} lines [{args.mode}]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
