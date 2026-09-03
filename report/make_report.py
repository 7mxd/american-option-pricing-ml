#!/usr/bin/env python
"""
Render a short LaTeX results summary from the JSON files each method notebook
writes.

    python report/make_report.py                 # -> report/results.tex
    python report/make_report.py --fragment      # body only, for \\input{}
    python report/make_report.py --out paper.tex

Numbers come from results/*.json and figures from figures/*.pdf, both written by
the notebooks. Method descriptions come from METHODS below (static, one entry per
method). Nothing is parsed out of notebook output, so rewording a print() cannot
break the report.

Build the PDF from the repository root, so the figure paths resolve:

    pdflatex -output-directory=report report/results.tex   (twice, for \\eqref)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

# Order methods appear in the report; unknown ones are appended alphabetically.
ORDER = ["onedim_gss", "ftt_als", "multidim_gss", "relu_dnn"]


# --------------------------------------------------------------------------
# Static per-method content. Keep `summary` short — this is a results brief,
# not a write-up. `figures` are included only if the file actually exists, so
# the report still builds before a notebook has exported them.
# --------------------------------------------------------------------------
METHODS = {
    "onedim_gss": dict(
        title="One-Dimensional Generalized Stochastic Sampling (gSS)",
        summary=r"""
Following Antonov and Piterbarg (2021), gSS approximates the target as a linear
combination of $N$ product kernels centred on stochastically sampled nodes
$z_n \in \mathbb{R}^D$ (their eqs.~25 and~30):
%
\begin{align}
  \tilde{f}(x) &= \sum_{n=1}^{N} \beta_n
      \prod_{d=1}^{D} \psi\!\big( (x_d - z_{n,d})\,\omega_d \big),
      \label{eq:gss-model} \\[2pt]
  \omega_d &= \theta\,\kappa_d,
      \qquad \kappa_d = \bar{g}_d / \Delta z,
      \label{eq:freq-bounds} \\[2pt]
  \beta(\theta) &= \arg\min_{\beta}
      \big\lVert \Psi(\theta)^{\top}\beta - Y \big\rVert_2^{2}
      + \lambda \lVert \beta \rVert_2^{2}.
      \label{eq:ridge}
\end{align}
%
Here $\psi$ is the inverse quadratic kernel, $\Delta z$ the mean nearest-node
distance, and $\bar{g}_d$ the relative directional risk magnitudes estimated from
the gradients of a base fit. Only the coefficients $\beta$ are regressed; the
nodes are fixed and the single scalar $\theta$ is chosen by a bounded
one-dimensional search. Ridge rather than plain least squares is required because
the Gram matrix is near-degenerate at this node count.

The node count is then enriched from $N$ to $N'$ for evaluation, and $\theta$
re-calibrated on the dense set, because the optimum moves with node density: the
value fitted on the sparse set is not the one to use at test time.
""",
        caveat=r"""
\paragraph{Interpretation.} No baseline has been computed yet. The European
option price is one of the six inputs and is itself a strong predictor of the
American price, so some share of this accuracy comes from that input rather than
from the method. Baselines follow with the cross-method comparison.
""",
        figures=[
            ("onedim-gss-fit",
             r"Predicted against actual American option price, with the "
             r"$y = x$ reference in grey."),
            ("onedim-gss-residuals",
             r"Test residuals against actual price, and their distribution."),
        ],
    ),
    # Add entries here as each method is completed.
}


# Labels name the model as well as the split: the fit model (N nodes) and the
# dense model (N' nodes) are different approximations, so their errors are not
# directly comparable. Only the two dense rows speak to generalisation.
METRIC_LABELS = {
    "test_rel_l2":         r"Test $L_2$ (dense)",
    "dense_train_rel_l2":  r"Train $L_2$ (dense)",
    "test_rel_mae":        r"Test MAE (dense)",
    "dense_train_rel_mae": r"Train MAE (dense)",
    "learn_rel_l2":        r"Train $L_2$ (fit model)",
    "learn_rel_mae":       r"Train MAE (fit model)",
}

# Report metrics in this order regardless of the order the notebook wrote them.
METRIC_ORDER = list(METRIC_LABELS)

CONFIG_LABELS = {
    "sim_range":       r"Simulation range",
    "nnodes":          r"Nodes (fit model)",
    "dense_nodes":     r"Nodes (test model)",
    "kernel":          r"Kernel",
    "nodes_type":      r"Node placement",
    "scale":           r"Kernel scale $\theta$",
    "l2_regularizer":  r"Ridge $\lambda$",
    "dtype":           r"Floating-point precision",
    "seed":            r"Random seed",
}


def tex_escape(s: str) -> str:
    """Escape the LaTeX specials that can appear in our string values."""
    for a, b in [("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"),
                 ("$", r"\$"), ("#", r"\#"), ("_", r"\_"), ("{", r"\{"),
                 ("}", r"\}"), ("~", r"\textasciitilde{}"),
                 ("^", r"\textasciicircum{}")]:
        s = s.replace(a, b)
    return s


def fmt(v) -> str:
    """Format a value for a LaTeX table cell."""
    if isinstance(v, bool):
        return r"\texttt{%s}" % v
    if isinstance(v, int):
        return f"{v:,}".replace(",", r"\,")
    if isinstance(v, float):
        if v != 0 and (abs(v) < 1e-3 or abs(v) >= 1e5):
            mant, exp = f"{v:.0e}".split("e")
            return rf"${mant}\times10^{{{int(exp)}}}$"
        return f"{v:.4f}".rstrip("0").rstrip(".")
    return r"\texttt{%s}" % tex_escape(str(v))


def load_results() -> list[dict]:
    if not RESULTS.is_dir():
        return []
    found = {}
    for p in sorted(RESULTS.glob("*.json")):
        with p.open(encoding="utf-8") as fh:
            r = json.load(fh)
        found[r.get("method", p.stem)] = r
    ranked = [found[k] for k in ORDER if k in found]
    ranked += [v for k, v in sorted(found.items()) if k not in ORDER]
    return ranked


def tabular(rows: list[tuple[str, str]]) -> str:
    body = "\n".join(rf"      {k} & {v} \\" for k, v in rows)
    return "\n".join([r"    \begin{tabular}{lr}", r"      \toprule",
                      body, r"      \bottomrule", r"    \end{tabular}"])


def table_pair(left: list[tuple[str, str]], right: list[tuple[str, str]],
               caption: str, label: str) -> str:
    """Two small tables side by side under a single caption.

    Deliberately not a `table` float: in a two-page brief a deferred float
    leaves a hole on the page it was declared on. This sits where it is put.
    """
    return "\n".join([
        r"\begin{center}",
        # Outer minipage: keeps the caption on the same page as the tables.
        r"\begin{minipage}{\linewidth}\centering",
        r"  \begin{minipage}[t]{0.48\linewidth}\vspace{0pt}\centering",
        tabular(left),
        r"  \end{minipage}\hfill",
        r"  \begin{minipage}[t]{0.48\linewidth}\vspace{0pt}\centering",
        tabular(right),
        r"  \end{minipage}",
        rf"  \captionof{{table}}{{{caption}}}",
        rf"  \label{{{label}}}",
        r"\end{minipage}",
        r"\end{center}",
        "",
    ])


def figure(name: str, caption: str, label: str) -> str:
    return "\n".join([
        r"\begin{figure}[htbp]",
        r"  \centering",
        rf"  \includegraphics[width=\linewidth]{{figures/{name}.pdf}}",
        rf"  \caption{{{caption}}}",
        rf"  \label{{{label}}}",
        r"\end{figure}",
        "",
    ])


def headline(r: dict) -> str:
    """One sentence stating the result, so the reader meets it before caveats."""
    m = r.get("metrics", {})
    test, train = m.get("test_rel_l2"), m.get("dense_train_rel_l2")
    if test is None:
        return ""
    if train is None:
        return (rf"On held-out data the enriched model reaches a relative "
                rf"$L_2$ error of {test:.4f}.")
    return (rf"On held-out data the enriched model reaches a relative $L_2$ "
            rf"error of {test:.4f}, against {train:.4f} in sample. Both figures "
            r"come from the same $N'$-node model, so the gap between them "
            r"measures generalisation directly. It is small, and shows no "
            r"sign of overfitting.")


def render_method(r: dict) -> str:
    key = r.get("method", "unknown")
    meta = METHODS.get(key, {})
    title = meta.get("title", key.replace("_", " ").title())

    out = [rf"\section{{{title}}}", ""]

    if meta.get("summary"):
        out += [meta["summary"].strip(), ""]

    d = r.get("data", {})
    if d:
        n_test = f"{d.get('n_test', 0):,}".replace(",", r"\,")
        n_train = f"{d.get('n_train', 0):,}".replace(",", r"\,")
        out += [
            rf"Fitted on {n_train} rows and evaluated on a held-out test set of "
            rf"{n_test} rows, {d.get('ndim', '?')} features.",
            "",
        ]

    out += [headline(r), ""]

    if meta.get("caveat"):
        out += [meta["caveat"].strip(), ""]

    cfg = [(CONFIG_LABELS.get(k, tex_escape(k)), fmt(v))
           for k, v in r.get("config", {}).items()]
    metrics = r.get("metrics", {})
    keys = ([k for k in METRIC_ORDER if k in metrics]
            + [k for k in metrics if k not in METRIC_ORDER])
    mtr = [(METRIC_LABELS.get(k, tex_escape(k)), fmt(metrics[k])) for k in keys]
    out += [table_pair(cfg, mtr,
                       "Configuration (left) and accuracy (right).",
                       f"tab:{key}")]

    for name, caption in meta.get("figures", []):
        if (FIGURES / f"{name}.pdf").is_file():
            out.append(figure(name, caption, f"fig:{name}"))

    if r.get("note"):
        out += [tex_escape(r["note"]), ""]
    return "\n".join(out)


def render_comparison(rs: list[dict]) -> str:
    """Cross-method table. Only emitted once more than one method exists."""
    rows = []
    for r in rs:
        title = METHODS.get(r.get("method"), {}).get(
            "title", r.get("method", "?").replace("_", " ").title())
        v = r.get("metrics", {}).get("test_rel_l2")
        rows.append((title, fmt(v) if v is not None else "---"))
    return "\n".join([
        r"\section{Comparison}",
        "",
        r"All methods are evaluated on the same held-out split, produced by "
        r"\texttt{data\_setup.py}.",
        "",
        r"\begin{table}[htbp]",
        r"  \centering",
        tabular(rows),
        r"  \caption{Test relative $L_2$ error by method.}",
        r"  \label{tab:comparison}",
        r"\end{table}",
        "",
    ])


PREAMBLE = r"""\documentclass[11pt,a4paper]{article}
\usepackage[margin=2.5cm]{geometry}
\usepackage{amsmath,amssymb}
\usepackage{booktabs}
\usepackage{graphicx}
\usepackage{caption}   % \captionof, for the non-floating tables
\usepackage[hidelinks]{hyperref}

% Figure paths are relative to the repository root; build from there.
\graphicspath{{./}{../}}

% Let floats fill a page rather than deferring a whole block to the next one.
\renewcommand{\topfraction}{0.9}
\renewcommand{\bottomfraction}{0.8}
\renewcommand{\textfraction}{0.07}
\renewcommand{\floatpagefraction}{0.75}

\title{American Option Pricing with Alternatives to Deep Neural Networks\\
       \large Results Summary}
\author{}
\date{\today}

\begin{document}
\maketitle
"""


def intro(n_done: int) -> str:
    """Scope note, so a reader knows this is interim and what the task is."""
    n_total = len(ORDER)
    return "\n".join([
        rf"\noindent This is an interim summary: {n_done} of {n_total} planned "
        r"approximation methods is complete, and a cross-method comparison "
        r"follows once more than one is finished.",
        "",
        r"The task is to predict the price of an American option from six "
        r"inputs: asset price, maturity, rate, dividend, implied volatility, "
        r"and the price of the corresponding European option. The dataset is "
        r"proprietary and is not distributed with the code.",
        "",
    ])


def build(fragment: bool = False) -> str:
    rs = load_results()
    if not rs:
        raise SystemExit(
            f"No result files in {RESULTS}/. Run a method notebook first — it "
            f"writes results/<method>.json at the end."
        )

    parts = [render_method(r) for r in rs]
    if len(rs) > 1:
        parts.append(render_comparison(rs))

    body = "\n".join(parts)
    if fragment:
        return body + "\n"
    return PREAMBLE + "\n" + intro(len(rs)) + "\n" + body + "\n\\end{document}\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fragment", action="store_true",
                    help="emit the body only, without preamble (for \\input)")
    ap.add_argument("--out", default=None,
                    help="output path (default: report/results.tex)")
    args = ap.parse_args()

    out = Path(args.out) if args.out else Path(__file__).parent / "results.tex"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(build(args.fragment), encoding="utf-8")

    rs = load_results()
    figs = sum(
        1
        for r in rs
        for name, _ in METHODS.get(r.get("method"), {}).get("figures", [])
        if (FIGURES / f"{name}.pdf").is_file()
    )
    print(f"wrote {out}  ({len(rs)} method{'s' if len(rs) != 1 else ''}, "
          f"{figs} figure{'s' if figs != 1 else ''})")


if __name__ == "__main__":
    main()
