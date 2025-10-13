#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
from typing import List, Tuple
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection
from matplotlib.patches import Rectangle
from matplotlib.widgets import Slider, Button
import math

# -------------------------------
# Data loaders (MovingAI formats)
# -------------------------------

def load_movingai_map(path: str) -> Tuple[int, int, List[List[int]]]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Map file not found: {path}")

    lines = p.read_text().splitlines()
    try:
        map_idx = next(i for i, ln in enumerate(lines) if ln.strip().lower() == "map")
    except StopIteration:
        raise ValueError("Invalid .map: missing 'map' header line.")

    raw_rows = [ln.rstrip("\n") for ln in lines[map_idx + 1:]]
    if not raw_rows:
        raise ValueError("Invalid .map: no grid lines after 'map'.")

    H_file = len(raw_rows)
    W_file = len(raw_rows[0])
    if any(len(r) != W_file for r in raw_rows):
        raise ValueError("Invalid .map: inconsistent row lengths.")

    free_file = [[0]*H_file for _ in range(W_file)]
    for y_file, row in enumerate(raw_rows):
        for x, ch in enumerate(row):
            free_file[x][y_file] = 1 if ch in ('.', 'G', 'S') else 0

    H = H_file
    W = W_file
    free = [[0]*H for _ in range(W)]
    for x in range(W):
        for y_file in range(H_file):
            y_plot = H - 1 - y_file
            free[x][y_plot] = free_file[x][y_file]

    return W, H, free


def load_movingai_scen(path: str, A: int, map_height: int) -> List[Tuple[int, int, int, int]]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Scenario file not found: {path}")

    lines = [ln.strip() for ln in p.read_text().splitlines() if ln.strip()]
    if not lines:
        raise ValueError("Empty .scen file.")

    start_idx = 0
    if lines[0].lower().startswith("version"):
        start_idx = 1

    agents: List[Tuple[int, int, int, int]] = []
    for ln in lines[start_idx:]:
        if len(agents) >= A:
            break
        parts = ln.split()
        if len(parts) < 8:
            raise ValueError(f"Malformed .scen line (need at least 8 columns): '{ln}'")

        try:
            mapW = int(parts[2])
            mapH = int(parts[3])
            sx   = int(parts[4])
            sy   = int(parts[5])
            gx   = int(parts[6])
            gy   = int(parts[7])
        except Exception as e:
            raise ValueError(f"Could not parse integer fields from .scen line: '{ln}' ({e})")

        if not (0 <= sx < mapW and 0 <= sy < mapH and 0 <= gx < mapW and 0 <= gy < mapH):
            raise ValueError(f"Coordinates out of bounds for map {mapW}x{mapH}: '{ln}'")

        sy_plot = map_height - 1 - sy
        gy_plot = map_height - 1 - gy
        agents.append((sx, sy_plot, gx, gy_plot))

    if len(agents) < A:
        raise ValueError(f"Scenario file has only {len(agents)} usable entries; need A={A}.")
    return agents


# ------------------------------------------
# Trajectory parsing
# ------------------------------------------

def parse_trajectories(path: str, A: int) -> List[List[List[Tuple[float, float]]]]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Trajectories file not found: {path}")
    lines = [ln.strip() for ln in p.read_text().splitlines() if ln.strip()]
    if not lines:
        raise ValueError("Trajectories file is empty.")

    if len(lines) % A != 0:
        raise ValueError(f"Line count {len(lines)} is not divisible by A={A}.")

    nIters = len(lines) // A
    iters: List[List[List[Tuple[float, float]]]] = []
    for i in range(nIters):
        iter_block: List[List[Tuple[float, float]]] = []
        for a in range(A):
            ln = lines[i*A + a]
            pts = []
            for chunk in ln.split(';'):
                chunk = chunk.strip()
                if not chunk:
                    continue
                if ',' not in chunk:
                    raise ValueError(f"Bad coord '{chunk}' in line {i*A+a+1}")
                xs, ys = chunk.split(',', 1)
                x = float(xs.strip())
                y = float(ys.strip())
                pts.append((x, y))
            iter_block.append(pts)
        iters.append(iter_block)
    return iters


# ------------------------------------------
# In-cell placement policy
# ------------------------------------------

def agent_cell_fraction(a: int, A: int) -> Tuple[float, float]:
    """
    Returns (fx, fy) inside the cell:
      A = 1 -> (1/2, 1/2)
      A = 2 -> (1/3,1/3), (2/3,2/3)
      A = 3 -> (1/3,1/3), (1/2,1/2), (2/3,2/3)
      A >= 4 -> evenly spaced along diagonal from 1/3 to 2/3
    """
    if A <= 0:
        return (0.5, 0.5)
    if A == 1:
        seq = [(0.5, 0.5)]
    elif A == 2:
        seq = [(1/3, 1/3), (2/3, 2/3)]
    elif A == 3:
        seq = [(1/3, 1/3), (0.5, 0.5), (2/3, 2/3)]
    else:
        # linspace from 1/3 to 2/3 with A points
        vals = np.linspace(1.0/3.0, 2.0/3.0, A)
        seq = [(float(v), float(v)) for v in vals]
    return seq[a % len(seq)]

def place_in_cell(x: float, y: float, a: int, A: int) -> Tuple[float, float]:
    """
    Map (x,y) to the same grid cell with the agent-specific fraction.
    """
    ix = math.floor(x)
    iy = math.floor(y)
    fx, fy = agent_cell_fraction(a, A)
    return ix + fx, iy + fy

def offset_traj(traj: List[Tuple[float, float]], a: int, A: int) -> List[Tuple[float, float]]:
    return [place_in_cell(x, y, a, A) for (x, y) in traj]


# ---------------
# Rendering / UI
# ---------------

def main():
    ap = argparse.ArgumentParser(
        description="Render per-iteration agent paths on MovingAI map; in-cell positions set per agent count."
    )
    ap.add_argument("map",  help=".map file (MovingAI ASCII grid)")
    ap.add_argument("scen", help=".scen file (MovingAI scenario list)")
    ap.add_argument("traj", help="trajectories file: A lines per iteration; 'x,y;x,y;...' per line")
    ap.add_argument("-A", "--agents", type=int, required=True, help="number of agents (first A scen entries; A lines per iteration)")
    ap.add_argument("--interval", type=int, default=800, help="ms between frames (iterations)")
    ap.add_argument("--save", help="Optional output filename (.mp4 or .gif)")
    args = ap.parse_args()

    # --- load inputs ---
    W, H, free = load_movingai_map(args.map)
    A = int(args.agents)
    agents = load_movingai_scen(args.scen, A, H)
    starts_raw = [(sx, sy) for (sx, sy, gx, gy) in agents]
    goals_raw  = [(gx, gy) for (sx, sy, gx, gy) in agents]
    iters  = parse_trajectories(args.traj, A)
    nIters = len(iters)

    # colors per agent (same for path, start, goal)
    cmap = plt.get_cmap("tab10")
    agent_colors = [cmap(i % 10) for i in range(A)]

    # --- figure/axes ---
    fig, ax = plt.subplots()
    plt.subplots_adjust(left=0.08, right=0.98, top=0.93, bottom=0.30)

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)

    # Show cell boundaries in light grey; hide labels to avoid overlap
    ax.set_xticks(np.arange(0, W + 1, 1))
    ax.set_yticks(np.arange(0, H + 1, 1))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.grid(True, which="both", color="lightgrey", linewidth=0.5)
    ax.set_title("")

    hud = fig.text(0.985, 0.965, "", ha="right", va="top", fontsize=10,
                   bbox=dict(boxstyle="round", facecolor="white", alpha=0.7, edgecolor="#ccc"))

    # Draw blocked cells; no edge so grid lines stay visible
    for x in range(W):
        for y in range(H):
            if free[x][y] == 0:
                ax.add_patch(Rectangle((x, y), 1, 1,
                                       facecolor="#555555", edgecolor="none",
                                       linewidth=0.0, zorder=0))

    # collections and markers (colored per agent)
    lcs = []
    for a in range(A):
        lc = LineCollection([], linewidths=2, zorder=2)
        lc.set_color(agent_colors[a])
        ax.add_collection(lc)
        lcs.append(lc)

    pts_start = [ax.plot([], [], marker="o", markersize=5,
                         color=agent_colors[a],
                         markeredgecolor="black", markeredgewidth=0.8,
                         linestyle="None", zorder=3)[0] for a in range(A)]
    pts_goal_inner = [ax.plot([], [], marker="o", markersize=5,
                              color=agent_colors[a],
                              markeredgecolor=agent_colors[a], markeredgewidth=0.8,
                              linestyle="None", zorder=4)[0] for a in range(A)]
    pts_goal_ring = [ax.plot([], [], marker="o", markersize=7, fillstyle="none",
                             markeredgecolor="black", markeredgewidth=1.0,
                             linestyle="None", zorder=5)[0] for _ in range(A)]

    # precompute in-cell start/goal placements
    starts = [place_in_cell(sx, sy, a, A) for a, (sx, sy) in enumerate(starts_raw)]
    goals  = [place_in_cell(gx, gy, a, A) for a, (gx, gy) in enumerate(goals_raw)]

    current_iter = 0
    paused = False

    def draw_iteration(i: int):
        it = iters[i]
        for a in range(A):
            traj = offset_traj(it[a], a, A)
            pts = np.asarray(traj, dtype=float)

            if len(pts) >= 2:
                segs = np.stack([pts[:-1], pts[1:]], axis=1)
                lcs[a].set_segments(segs)
            else:
                lcs[a].set_segments([])

            sx, sy = starts[a]
            pts_start[a].set_data([sx], [sy])

            gx, gy = goals[a]
            pts_goal_inner[a].set_data([gx], [gy])
            pts_goal_ring[a].set_data([gx], [gy])

        hud.set_text(f"iter: {i + 1} / {nIters}")
        fig.canvas.draw_idle()

    def init():
        draw_iteration(current_iter)

    def update(frame):
        nonlocal current_iter
        if not paused:
            current_iter = frame % max(1, nIters)
            slider.eventson = False
            slider.set_val(current_iter)
            slider.eventson = True
            draw_iteration(current_iter)

    anim = FuncAnimation(fig, update, frames=max(1, nIters), init_func=init,
                         interval=max(1, args.interval), blit=False, repeat=True)

    # --- slider + buttons ---
    axpos = ax.get_position()
    slider_h = 0.045
    slider_gap = 0.11
    slider_y = axpos.y0 - slider_gap
    ax_slide = plt.axes([axpos.x0, slider_y, axpos.width, slider_h])
    slider = Slider(ax_slide, "Iteration", 0, max(0, nIters - 1), valinit=current_iter, valstep=1)

    btn_h = 0.06
    btn_w = 0.12
    btn_gap = 0.03
    group_w = 3 * btn_w + 2 * btn_gap
    group_left = axpos.x0 + (axpos.width - group_w) / 2.0
    btn_y = slider_y - (btn_h + 0.025)

    ax_prev = plt.axes([group_left + 0 * (btn_w + btn_gap), btn_y, btn_w, btn_h])
    ax_play = plt.axes([group_left + 1 * (btn_w + btn_gap), btn_y, btn_w, btn_h])
    ax_next = plt.axes([group_left + 2 * (btn_w + btn_gap), btn_y, btn_w, btn_h])

    btn_prev = Button(ax_prev, "Prev")
    btn_next = Button(ax_next, "Next")
    btn_play = Button(ax_play, "Pause")

    def do_prev(event=None):
        nonlocal paused, current_iter
        paused = True
        btn_play.label.set_text("Play")
        current_iter = max(0, current_iter - 1)
        slider.set_val(current_iter)

    def do_next(event=None):
        nonlocal paused, current_iter
        paused = True
        btn_play.label.set_text("Play")
        current_iter = min(max(0, nIters - 1), current_iter + 1)
        slider.set_val(current_iter)

    def do_play(event=None):
        nonlocal paused
        paused = not paused
        btn_play.label.set_text("Pause" if not paused else "Play")

    def on_slide(val):
        nonlocal current_iter, paused
        paused = True
        btn_play.label.set_text("Play")
        current_iter = int(val)
        draw_iteration(current_iter)

    btn_prev.on_clicked(do_prev)
    btn_next.on_clicked(do_next)
    btn_play.on_clicked(do_play)
    slider.on_changed(on_slide)

    # --- output ---
    if args.save:
        ext = args.save.lower().split(".")[-1]
        fps = max(1, 1000 // max(1, args.interval))
        try:
            if ext == "mp4":
                anim.save(args.save, writer="ffmpeg", fps=fps)
            elif ext == "gif":
                anim.save(args.save, writer="imagemagick", fps=fps)
            else:
                print("Unknown extension; use .mp4 or .gif. Showing window instead.")
                plt.show()
        except Exception as e:
            print(f"Save failed: {e}\nFalling back to interactive window.")
            plt.show()
    else:
        plt.show()


if __name__ == "__main__":
    main()

