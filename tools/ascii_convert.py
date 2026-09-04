#!/usr/bin/env python3
"""Convert docs/brand/ouro-logo.svg to terminal ASCII art.

The roundness law: terminal cells are ~1 wide x 2 tall, so art only
looks round when CHARS_WIDE == 2 * LINES_TALL. CELL_ASPECT encodes the
cell shape; every output grid derives its height from its width through
it — no converter may map the logo into a square character grid again
(that is how the tall-oval ascii-logo-trimmed.txt happened).

Modes (binary glyph sets have no half-tones: thin ring gaps and the
head/eye vanish — that is what the ramp and dither options fix):

  solid  one coverage sample per cell -> '█' / space (binary)
  ramp   coverage -> density glyph from RAMP (e.g. ' ░▒▓█') — the
         half-value mode: thin gaps render as ░/▒ instead of vanishing
  half   two pixel-rows per line -> '▀' / '▄' / '█' / space (crisper
         curves at the same physical size: W x 2H real pixels)

Options:
  --dither bayer   ordered 4x4 dithering for solid/half: binary glyphs,
                   spatial density carries the half-tones
  --gamma G        coverage ** G before rendering (<1 lifts mid-tones)
  --threshold T    ink cutoff for binary modes (lower keeps thin gaps)

Usage:
  ascii_convert.py --width 120 --mode ramp --out docs/brand/ascii-logo.txt
"""
import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

from PIL import Image

CELL_ASPECT = 0.5  # char width / char height on a classic terminal
SUPER = 4          # supersampling factor per cell
DEFAULT_THRESHOLD = 0.28  # binary ink cutoff (low: thin gaps survive)
DEFAULT_RAMP = " ░▒▓█"    # lightest -> fullest

# ordered 4x4 Bayer matrix, normalized to (b + 0.5) / 16
BAYER4 = [
    [0, 8, 2, 10],
    [12, 4, 14, 6],
    [3, 11, 1, 9],
    [15, 7, 13, 5],
]


def rasterize(svg: Path, px: int) -> Image.Image:
    with tempfile.NamedTemporaryFile(suffix=".png") as tmp:
        subprocess.run(
            ["rsvg-convert", "-w", str(px), "-h", str(px), str(svg), "-o", tmp.name],
            check=True,
        )
        return Image.open(tmp.name).convert("RGBA")


def coverage_grid(svg: Path, width: int, pixel_rows: int, gamma: float) -> list[list[float]]:
    """Alpha coverage per cell in [0, 1], gamma-shaped, cropped to the
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

    grid = [[0.0] * width for _ in range(pixel_rows)]
    for r in range(pixel_rows):
        for c in range(width):
            total = 0
            for y in range(r * SUPER, (r + 1) * SUPER):
                row_off = y * ss_w
                for x in range(c * SUPER, (c + 1) * SUPER):
                    total += data[row_off + x]
            cov = total / (SUPER * SUPER * 255)
            grid[r][c] = cov**gamma if gamma != 1.0 else cov
    return grid


def bits_from(cov: list[list[float]], threshold: float, dither: str) -> list[list[int]]:
    if dither != "bayer":
        return [[1 if c >= threshold else 0 for c in row] for row in cov]
    return [
        [1 if c > (BAYER4[y % 4][x % 4] + 0.5) / 16 else 0 for x, c in enumerate(row)]
        for y, row in enumerate(cov)
    ]


def emit_solid(bits: list[list[int]]) -> str:
    return "\n".join(
        "".join("█" if cell else " " for cell in row).rstrip() for row in bits
    )


def emit_ramp(cov: list[list[float]], ramp: str) -> str:
    n = len(ramp)
    return "\n".join(
        "".join(ramp[min(int(c * n), n - 1)] for c in row).rstrip() for row in cov
    )


HALF_GLYPH = {(1, 1): "█", (1, 0): "▀", (0, 1): "▄", (0, 0): " "}


def emit_half(bits: list[list[int]]) -> str:
    # bits has 2*lines pixel rows; pair them per output line
    lines = []
    for r in range(0, len(bits), 2):
        top, bot = bits[r], bits[r + 1]
        lines.append(
            "".join(HALF_GLYPH[(t, b)] for t, b in zip(top, bot)).rstrip()
        )
    return "\n".join(lines)


def render(args, cov: list[list[float]]) -> str:
    if args.mode == "ramp":
        return emit_ramp(cov, args.ramp)
    bits = bits_from(cov, args.threshold, args.dither)
    return emit_solid(bits) if args.mode == "solid" else emit_half(bits)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--svg", type=Path, default=Path("docs/brand/ouro-logo.svg"))
    ap.add_argument("--width", type=int, required=True, help="chars wide")
    ap.add_argument("--mode", choices=["solid", "ramp", "half"], required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    ap.add_argument("--dither", choices=["none", "bayer"], default="none")
    ap.add_argument("--gamma", type=float, default=1.0)
    ap.add_argument("--ramp", default=DEFAULT_RAMP)
    args = ap.parse_args()

    lines = args.width // 2  # the roundness law: chars_wide == 2 * lines
    pixel_rows = lines if args.mode in ("solid", "ramp") else lines * 2
    cov = coverage_grid(args.svg, args.width, pixel_rows, args.gamma)

    text = render(args, cov)
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
