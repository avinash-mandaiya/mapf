#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Render agent trajectories from a run_id by automatically loading parameters from .stats file.

Usage:
    python render/render_all2.py <run_id> [OPTIONS]

Basic Examples:
    python render/render_all2.py m1
    python render/render_all2.py m1 --interval 400
    python render/render_all2.py m1 --save output.mp4
    python render/render_all2.py m1 --save animation.gif

Mode Examples:
    python render/render_all2.py m1 --mode map                    # Default: map only
    python render/render_all2.py m1 --mode map+error              # Map with error plot side-by-side
    python render/render_all2.py m1 --mode map+error --error-cols 1  # Show only total error (column 1, 0-indexed)
    python render/render_all2.py m1 --mode map+error --error-cols 1,2,3  # Show multiple error columns

Combined Examples:
    python render/render_all2.py m1 --interval 400 --mode map+error
    python render/render_all2.py m1 --mode map+error --error-cols 1,2,3 --save animation.mp4

Options:
    --interval MILLISECONDS  : Time between frames (default: 200)
    --save FILENAME         : Save animation as .mp4 or .gif
    --mode {map,map+error}  : Render mode (default: map)
    --error-cols COLS       : Comma-separated column indices for error plot (0-indexed, default: 1 for total error)

Interactive Controls:
    - Prev/Play/Next buttons: Control animation playback
    - Slider: Jump to specific frame
    - Save Frame button: Save current map frame as PNG
    - Save Error button (map+error mode only): Save full error plot as PNG

The script will:
- Read <run_id>.stats to get the .map file, .scen file, and number of agents
- Read <run_id>.sol for the trajectory data
- Read <run_id>.err for the error data (in map+error mode)
- Display an interactive visualization with playback controls
"""

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
# Stats file parser
# ------------------------------------------

def parse_stats_file(stats_path: str) -> Tuple[str, str, int]:
    """
    Parse the .stats file to extract map path, scen path, and number of agents.
    
    Expected format:
    ./mapf <map_file> <scen_file> <num_agents> ...
    
    Returns:
        (map_path, scen_path, num_agents)
    """
    p = Path(stats_path)
    if not p.exists():
        raise FileNotFoundError(f"Stats file not found: {stats_path}")
    
    lines = [ln.strip() for ln in p.read_text().splitlines() if ln.strip()]
    if not lines:
        raise ValueError(f"Stats file is empty: {stats_path}")
    
    parts = lines[0].split()
    if len(parts) < 4:
        raise ValueError(f"Stats file malformed. Expected at least 4 fields, got {len(parts)}")
    
    map_path = parts[1]
    scen_path = parts[2]
    try:
        num_agents = int(parts[3])
    except ValueError:
        raise ValueError(f"Could not parse number of agents from '{parts[3]}'")
    
    return map_path, scen_path, num_agents


# ------------------------------------------
# Error file parser
# ------------------------------------------

def load_error_file(path: str) -> np.ndarray:
    """
    Load error data from file.
    
    Expected format: space/tab separated columns
    Column 0: iteration number
    Column 1+: error values
    
    Returns:
        2D numpy array of shape (n_frames, n_columns)
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Error file not found: {path}")
    
    data = np.loadtxt(path)
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    
    return data


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

    nFrames = len(lines) // A
    frames: List[List[List[Tuple[float, float]]]] = []
    for i in range(nFrames):
        frame_block: List[List[Tuple[float, float]]] = []
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
            frame_block.append(pts)
        frames.append(frame_block)
    return frames


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
        description="Render per-frame agent paths from run_id; automatically loads config from .stats file."
    )
    ap.add_argument("run_id", help="Run identifier (will load <run_id>.stats and <run_id>.sol)")
    ap.add_argument("--interval", type=int, default=200, help="ms between frames (default: 200)")
    ap.add_argument("--save", help="Optional output filename for animation (.mp4 or .gif)")
    ap.add_argument("--mode", choices=["map", "map+error"], default="map",
                    help="Render mode: 'map' (default) or 'map+error' (side-by-side with error plot)")
    ap.add_argument("--error-cols", type=str, default="1",
                    help="Comma-separated column indices for error plot (0-indexed, default: '1' for total error)")
    args = ap.parse_args()

    # Parse error columns
    error_cols = [int(c.strip()) for c in args.error_cols.split(',')]

    # --- parse stats file ---
    stats_file = f"{args.run_id}.stats"
    map_path, scen_path, A = parse_stats_file(stats_file)
    
    print(f"Loading configuration from {stats_file}")
    print(f"  Map file: {map_path}")
    print(f"  Scenario file: {scen_path}")
    print(f"  Number of agents: {A}")
    
    # --- load inputs ---
    W, H, free = load_movingai_map(map_path)
    agents = load_movingai_scen(scen_path, A, H)
    starts_raw = [(sx, sy) for (sx, sy, gx, gy) in agents]
    goals_raw  = [(gx, gy) for (sx, sy, gx, gy) in agents]
    
    traj_file = f"{args.run_id}.sol"
    frames = parse_trajectories(traj_file, A)
    nFrames = len(frames)
    
    print(f"  Trajectory file: {traj_file}")
    print(f"  Number of frames: {nFrames}")
    print(f"  Render mode: {args.mode}")

    # Load error data if in map+error mode
    error_data = None
    if args.mode == "map+error":
        error_file = f"{args.run_id}.err"
        error_data = load_error_file(error_file)
        print(f"  Error file: {error_file}")
        print(f"  Error columns to plot: {error_cols}")
        
        # Validate error columns
        max_col = error_data.shape[1] - 1
        for col in error_cols:
            if col < 0 or col > max_col:
                raise ValueError(f"Error column {col} out of range [0, {max_col}]")
        
        # Verify frame count matches
        if error_data.shape[0] != nFrames:
            raise ValueError(f"Frame count mismatch: {nFrames} frames in .sol but {error_data.shape[0]} rows in .err")

    # colors per agent (same for path, start, goal)
    cmap = plt.get_cmap("tab10")
    agent_colors = [cmap(i % 10) for i in range(A)]

    # --- figure/axes setup based on mode ---
    if args.mode == "map":
        fig, ax_map = plt.subplots(figsize=(10, 8))
        plt.subplots_adjust(left=0.08, right=0.98, top=0.96, bottom=0.30)
        ax_error = None
    else:  # map+error
        fig, (ax_map, ax_error) = plt.subplots(1, 2, figsize=(16, 8))
        plt.subplots_adjust(left=0.08, right=0.95, top=0.96, bottom=0.25, wspace=0.30)

    ax_map.set_aspect("equal", adjustable="box")
    ax_map.set_xlim(0, W)
    ax_map.set_ylim(0, H)

    # Show cell boundaries in light grey; only internal grid lines, ticks inside
    ax_map.set_xticks(np.arange(1, W, 1))
    ax_map.set_yticks(np.arange(1, H, 1))
    ax_map.set_xticklabels([])
    ax_map.set_yticklabels([])
    ax_map.tick_params(direction='in', length=0)  # Hide tick marks
    ax_map.grid(True, which="both", color="lightgrey", linewidth=0.5)

    # Draw blocked cells; no edge so grid lines stay visible
    for x in range(W):
        for y in range(H):
            if free[x][y] == 0:
                ax_map.add_patch(Rectangle((x, y), 1, 1,
                                       facecolor="#555555", edgecolor="none",
                                       linewidth=0.0, zorder=0))

    # collections and markers (colored per agent)
    lcs = []
    for a in range(A):
        lc = LineCollection([], linewidths=2, zorder=2)
        lc.set_color(agent_colors[a])
        ax_map.add_collection(lc)
        lcs.append(lc)

    # Start markers: circles
    pts_start = [ax_map.plot([], [], marker="o", markersize=5,
                         color=agent_colors[a],
                         markeredgecolor="black", markeredgewidth=0.8,
                         linestyle="None", zorder=3)[0] for a in range(A)]
    
    # Goal markers: stars
    pts_goal = [ax_map.plot([], [], marker="*", markersize=10,
                            color=agent_colors[a],
                            markeredgecolor="black", markeredgewidth=0.8,
                            linestyle="None", zorder=4)[0] for a in range(A)]

    # precompute in-cell start/goal placements
    starts = [place_in_cell(sx, sy, a, A) for a, (sx, sy) in enumerate(starts_raw)]
    goals  = [place_in_cell(gx, gy, a, A) for a, (gx, gy) in enumerate(goals_raw)]

    # --- Error plot setup ---
    error_lines = []
    if ax_error is not None:
        ax_error.set_yscale('log')
        ax_error.set_xlabel('Iteration', fontsize=12, fontweight='bold')
        ax_error.set_ylabel('Gap', fontsize=12, fontweight='bold')
        ax_error.set_title('Gap between constraint set A and B', fontsize=13, fontweight='bold')
        ax_error.tick_params(labelsize=10)
        
        # Create line objects for each error column
        error_plot_colors = plt.cm.tab10(np.linspace(0, 1, len(error_cols)))
        for i, col in enumerate(error_cols):
            line, = ax_error.plot([], [], '-o', linewidth=0.5, markersize=1,
                                 color=error_plot_colors[i],
                                 label=f'Error Col {col}' if len(error_cols) > 1 else 'Total Error')
            error_lines.append(line)
        
        if len(error_cols) > 1:
            ax_error.legend(loc='best', fontsize=10, framealpha=0.9)
        
        # Set x-axis limits based on actual iteration numbers
        min_iter = error_data[0, 0]
        max_iter = error_data[-1, 0]
        ax_error.set_xlim(min_iter, max_iter)
        
        # Set y-axis limits based on data
        min_err = np.min(error_data[:, error_cols])
        max_err = np.max(error_data[:, error_cols])
        margin = 0.2
        ax_error.set_ylim(min_err * 10**(-margin), max_err * 10**(margin))

    current_frame = 0
    paused = False

    def draw_frame(frame_idx: int):
        frame_data = frames[frame_idx]
        for a in range(A):
            traj = offset_traj(frame_data[a], a, A)
            pts = np.asarray(traj, dtype=float)

            if len(pts) >= 2:
                segs = np.stack([pts[:-1], pts[1:]], axis=1)
                lcs[a].set_segments(segs)
            else:
                lcs[a].set_segments([])

            sx, sy = starts[a]
            pts_start[a].set_data([sx], [sy])

            gx, gy = goals[a]
            pts_goal[a].set_data([gx], [gy])

        # Update error plot (show data up to current frame)
        if ax_error is not None and error_data is not None:
            for line_idx, col in enumerate(error_cols):
                x_data = error_data[:frame_idx+1, 0]  # Iteration numbers from column 0
                y_data = error_data[:frame_idx+1, col]  # Error values
                error_lines[line_idx].set_data(x_data, y_data)

        fig.canvas.draw_idle()

    def init():
        draw_frame(current_frame)

    def update(frame):
        nonlocal current_frame
        if not paused:
            current_frame = frame % max(1, nFrames)
            slider.eventson = False
            slider.set_val(current_frame)
            slider.eventson = True
            draw_frame(current_frame)

    anim = FuncAnimation(fig, update, frames=max(1, nFrames), init_func=init,
                         interval=max(1, args.interval), blit=False, repeat=True)

    # --- slider + buttons ---
    # Calculate position based on the figure
    if args.mode == "map":
        axpos = ax_map.get_position()
    else:
        # For map+error, center controls under both plots
        axpos_left = ax_map.get_position()
        axpos_right = ax_error.get_position()
        # Create a pseudo-position that spans both
        class PseudoPos:
            def __init__(self, left, right):
                self.x0 = left.x0
                self.width = right.x1 - left.x0
                self.y0 = left.y0
        axpos = PseudoPos(axpos_left, axpos_right)
    
    slider_h = 0.045
    slider_gap = 0.11
    slider_y = axpos.y0 - slider_gap
    ax_slide = plt.axes([axpos.x0, slider_y, axpos.width, slider_h])
    slider = Slider(ax_slide, "Frame", 0, max(0, nFrames - 1), valinit=current_frame, valstep=1)

    # Button layout - add extra button for map+error mode
    if args.mode == "map":
        n_buttons = 4
    else:
        n_buttons = 5
    
    btn_h = 0.06
    btn_w = 0.10
    btn_gap = 0.02
    group_w = n_buttons * btn_w + (n_buttons - 1) * btn_gap
    group_left = axpos.x0 + (axpos.width - group_w) / 2.0
    btn_y = slider_y - (btn_h + 0.025)

    ax_prev = plt.axes([group_left + 0 * (btn_w + btn_gap), btn_y, btn_w, btn_h])
    ax_play = plt.axes([group_left + 1 * (btn_w + btn_gap), btn_y, btn_w, btn_h])
    ax_next = plt.axes([group_left + 2 * (btn_w + btn_gap), btn_y, btn_w, btn_h])
    ax_save = plt.axes([group_left + 3 * (btn_w + btn_gap), btn_y, btn_w, btn_h])

    btn_prev = Button(ax_prev, "Prev")
    btn_next = Button(ax_next, "Next")
    btn_play = Button(ax_play, "Pause")
    btn_save = Button(ax_save, "Save Frame")

    if args.mode == "map+error":
        ax_save_err = plt.axes([group_left + 4 * (btn_w + btn_gap), btn_y, btn_w, btn_h])
        btn_save_err = Button(ax_save_err, "Save Error")

    def do_prev(event=None):
        nonlocal paused, current_frame
        paused = True
        btn_play.label.set_text("Play")
        current_frame = max(0, current_frame - 1)
        slider.set_val(current_frame)

    def do_next(event=None):
        nonlocal paused, current_frame
        paused = True
        btn_play.label.set_text("Play")
        current_frame = min(max(0, nFrames - 1), current_frame + 1)
        slider.set_val(current_frame)

    def do_play(event=None):
        nonlocal paused
        paused = not paused
        btn_play.label.set_text("Pause" if not paused else "Play")

    def on_slide(val):
        nonlocal current_frame, paused
        paused = True
        btn_play.label.set_text("Play")
        current_frame = int(val)
        draw_frame(current_frame)

    def do_save_frame(event=None):
        """Save the current frame showing only the map area"""
        filename = f"{args.run_id}_frame_{current_frame + 1}.png"
        
        try:
            # Save only the map axes
            extent = ax_map.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
            fig.savefig(filename, bbox_inches=extent.expanded(1.1, 1.1), dpi=150)
            print(f"Frame {current_frame + 1} saved to: {filename}")
        except Exception as e:
            print(f"Error saving frame: {e}")

    def do_save_error(event=None):
        """Save the error plot with all data points"""
        if ax_error is None:
            return
        
        filename = f"{args.run_id}_error.png"
        
        try:
            # Temporarily update error plot to show all data
            for line_idx, col in enumerate(error_cols):
                x_data = error_data[:, 0]  # All iteration numbers
                y_data = error_data[:, col]  # All error values
                error_lines[line_idx].set_data(x_data, y_data)
            
            fig.canvas.draw_idle()
            
            # Save only the error axes
            extent = ax_error.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
            fig.savefig(filename, bbox_inches=extent.expanded(1.15, 1.15), dpi=150)
            print(f"Error plot saved to: {filename}")
            
            # Restore error plot to current frame
            draw_frame(current_frame)
        except Exception as e:
            print(f"Error saving error plot: {e}")

    btn_prev.on_clicked(do_prev)
    btn_next.on_clicked(do_next)
    btn_play.on_clicked(do_play)
    btn_save.on_clicked(do_save_frame)
    if args.mode == "map+error":
        btn_save_err.on_clicked(do_save_error)
    slider.on_changed(on_slide)

    # --- output ---
    if args.save:
        ext = args.save.lower().split(".")[-1]
        fps = max(1, 1000 // max(1, args.interval))
        try:
            if ext == "mp4":
                anim.save(args.save, writer="ffmpeg", fps=fps)
                print(f"Animation saved to: {args.save}")
            elif ext == "gif":
                anim.save(args.save, writer="imagemagick", fps=fps)
                print(f"Animation saved to: {args.save}")
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
