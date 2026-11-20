"""Write a table as CSV and as a booktabs LaTeX tabular."""
from pathlib import Path

import pandas as pd


def tex(s):
    """LaTeX-safe text: the beta sign and the characters & % _ #."""
    s = str(s).replace("β", r"$\beta$")
    for ch in "&%_#":
        s = s.replace(ch, "\\" + ch)
    return s


def write(out_dir, name, header, rows, tex_header=None, tex_rows=None):
    """Write name.csv and name.tex into out_dir; return both paths.

    header, rows: plain cells. tex_header, tex_rows: ready LaTeX cells (default: the plain
    cells through tex()). First column left-aligned, the others centred."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=header).to_csv(out / f"{name}.csv", index=False, encoding="utf-8")
    tex_header = tex_header or [tex(h) for h in header]
    tex_rows = tex_rows or [[tex(c) for c in r] for r in rows]
    lines = [r"\begin{tabular}{l" + "c" * (len(header) - 1) + "}", r"\toprule",
             " & ".join(tex_header) + r" \\", r"\midrule",
             *[" & ".join(r) + r" \\" for r in tex_rows], r"\bottomrule", r"\end{tabular}"]
    (out / f"{name}.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return [out / f"{name}.csv", out / f"{name}.tex"]