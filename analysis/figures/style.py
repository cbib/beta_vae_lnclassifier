"""Journal style for every figure: Bioinformatics widths, triframe palette, one save()."""
import io
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image

MM = 1 / 25.4
WIDTH = {"single": 86 * MM, "double": 178 * MM}  # journal column widths, inches
INK, MUTED, HAIR, ACCENT = "#1d232b", "#68737f", "#dde2e7", "#d55e00"
RAMP = LinearSegmentedColormap.from_list("flag", ["#eef3f7", "#a9c4da", "#5b88ad", "#1f4b6e"])
FONTS = ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"]


def apply():
    """Set the global rcParams. Idempotent."""
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": FONTS, "font.size": 6.5,
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "text.color": INK, "axes.labelcolor": INK, "axes.edgecolor": INK,
        "xtick.color": INK, "ytick.color": INK,
        "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "axes.spines.top": False, "axes.spines.right": False,
        "legend.frameon": False, "figure.dpi": 150, "savefig.facecolor": "white",
    })


def resolved_font():
    """Name of the font matplotlib actually uses (recorded in the manifest)."""
    return font_manager.FontProperties(family=FONTS).get_name()


def new_figure(width, height_mm, nrows=1, ncols=1, **kw):
    """(fig, axes) at a journal width ("single" or "double") and a height in mm."""
    apply()
    return plt.subplots(nrows, ncols, figsize=(WIDTH[width], height_mm * MM),
                        layout="constrained", **kw)


def panel_label(ax, letter):
    """Bold panel letter at the top-left corner of ax."""
    ax.text(-0.02, 1.04, letter, transform=ax.transAxes, fontsize=8,
            fontweight="bold", ha="right", va="bottom")


def save(fig, out_dir, name, caption=None, alt=None, tiff_dpi=350):
    """Write name.pdf, name.png, name.tiff (RGB, LZW) and, if given, name_caption.txt.

    Never uses bbox_inches="tight", so the files keep the journal width.
    Returns the list of paths written."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = [out / f"{name}.pdf", out / f"{name}.png", out / f"{name}.tiff"]
    fig.savefig(paths[0])
    fig.savefig(paths[1], dpi=240)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=tiff_dpi)
    buf.seek(0)
    Image.open(buf).convert("RGB").save(paths[2], dpi=(tiff_dpi, tiff_dpi), compression="tiff_lzw")
    if caption or alt:
        text = (caption or "") + (f"\n\nAlt text: {alt}" if alt else "") + "\n"
        paths.append(out / f"{name}_caption.txt")
        paths[-1].write_text(text, encoding="utf-8")
    plt.close(fig)
    return paths