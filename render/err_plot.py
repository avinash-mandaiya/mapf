#!/usr/bin/env python3
import sys
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import itertools

def main(filepath, cols_to_plot):
    """
    Plot selected columns (after iteration) from a .err file on semilog-y scale.
    Usage:
        python err_plot2.py file.err [col1 col2 col3 ...]
    If no columns given, defaults to column 1 (second column in file).
    """
    path = Path(filepath)
    if not path.exists():
        print(f"Error: file '{path}' not found.")
        sys.exit(1)

    # ---- Style ----
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 12,
        "axes.labelsize": 13,
        "axes.linewidth": 1.0,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "figure.figsize": (10.5, 5.3),
        "savefig.dpi": 300,
        "savefig.bbox": "tight"
    })

    # ---- Read file ----
    data = pd.read_csv(path, delim_whitespace=True, comment="#", header=None)
    if data.shape[0] < 2 or data.shape[1] < 2:
        print("Error: file must have at least two columns and >1 row.")
        sys.exit(1)

    # Skip first row (manual type)
    data = data.iloc[1:, :]

    iterations = data.iloc[:, 0].astype(float)
    total_columns = data.shape[1] - 1

    # Default: plot column 1 if none specified
    if not cols_to_plot:
        cols_to_plot = [1]

    # Validate requested columns
    valid_cols = []
    for c in cols_to_plot:
        if 1 <= c <= total_columns:
            valid_cols.append(c)
        else:
            print(f"Warning: column {c} out of range; skipping.")

    if not valid_cols:
        print("No valid columns to plot.")
        sys.exit(1)

    # ---- Plot ----
    fig, ax = plt.subplots()
    colors = itertools.cycle(plt.cm.tab10.colors)

    for c in valid_cols:
        y = data.iloc[:, c].astype(float)
        ax.semilogy(
            iterations, y,
            lw=1.0, marker='o', markersize=0.8,
            markerfacecolor='white', markeredgewidth=0.6,
            color=next(colors),
            label=f"Column {c}"
        )

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Value (log scale)")
    ax.legend(frameon=False, loc='best')
    ax.grid(False)
    plt.tight_layout()
    plt.show()
    # fig.savefig(path.with_suffix(".pdf"))

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <file.err> [col1 col2 col3 ...]")
        sys.exit(1)

    filepath = sys.argv[1]
    cols = list(map(int, sys.argv[2:])) if len(sys.argv) > 2 else []
    main(filepath, cols)

