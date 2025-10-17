# pacman_mapf.py
# Usage:
#   INTERACTIVE (with buttons & slider):
#     python pacman_mapf.py <map.map> <scen.scen> <traj.sol> -A <agents> --interval 600
#   SAVE (no UI widgets):
#     MPLBACKEND=Agg python pacman_mapf.py <map.map> <scen.scen> <traj.sol> -A <agents> --interval 600 --save out.mp4

import argparse
from typing import List, Tuple
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button, Slider
from matplotlib.patches import Rectangle, Wedge, Circle
from pathlib import Path
import os
import random

# ---------- MovingAI parsers (map + scen) ----------

def parse_map(map_path: str) -> Tuple[int, int, List[Tuple[int,int]]]:
    p = Path(map_path)
    if not p.is_absolute():
        p = Path(__file__).resolve().parent.parent / p
    txt = p.read_text().splitlines()

    header = {}
    grid_start = None
    for i, line in enumerate(txt):
        line = line.strip()
        low = line.lower()
        if low.startswith("height"):
            header["height"] = int(line.split()[1])
        elif low.startswith("width"):
            header["width"] = int(line.split()[1])
        elif low == "map":
            grid_start = i + 1
            break

    if grid_start is None or "height" not in header or "width" not in header:
        raise ValueError("Unrecognized .map format (missing width/height/map)")

    H = header["height"]
    W = header["width"]
    rows = [list(row.rstrip("\n")) for row in txt[grid_start:grid_start+H]]
    if len(rows) != H or any(len(r) < W for r in rows):
        raise ValueError("Map grid size mismatch")

    obstacles = []
    for r in range(H):
        for c in range(W):
            ch = rows[r][c] if c < len(rows[r]) else '@'
            blocked = (ch != '.' and ch != 'G')
            if blocked:
                x = c
                y = (H - 1 - r)  # bottom-left origin
                obstacles.append((x, y))
    return H, W, obstacles

def parse_scen(scen_path: str, A: int) -> Tuple[List[Tuple[int,int]], List[Tuple[int,int]], int, int]:
    p = Path(scen_path)
    if not p.is_absolute():
        p = Path(__file__).resolve().parent.parent / p
    lines = [ln.strip() for ln in p.read_text().splitlines() if ln.strip()]

    idx = 0
    if lines and lines[0].lower().startswith("version"):
        idx = 1

    starts, goals = [], []
    width = height = None
    for ln in lines[idx:idx+A]:
        parts = ln.split()
        if len(parts) < 9:
            raise ValueError(f"Bad .scen line: {ln}")
        width = int(parts[2]); height = int(parts[3])
        sx = int(parts[4]); sy = int(parts[5])
        gx = int(parts[6]); gy = int(parts[7])
        sy_bl = height - 1 - sy
        gy_bl = height - 1 - gy
        starts.append((sx, sy_bl))
        goals.append((gx, gy_bl))
    if len(starts) < A:
        raise ValueError(f".scen file has only {len(starts)} tasks; need {A}")
    return starts, goals, width, height

def parse_instance(map_path: str, scen_path: str, A: int):
    rows, cols, obstacles = parse_map(map_path)
    starts, goals, _, _ = parse_scen(scen_path, A)
    return rows, cols, obstacles, A, starts, goals

# ---------- original helpers (unchanged logic) ----------

def parse_traj_line(s: str) -> List[Tuple[float, float]]:
    pts = []
    for seg in s.split(';'):
        seg = seg.strip()
        if not seg:
            continue
        parts = [t.strip() for t in seg.split(',') if t.strip()]
        if len(parts) != 2:
            raise ValueError(f"Bad trajectory segment: '{seg}'")
        x, y = float(parts[0]), float(parts[1])
        pts.append((x, y))
    if not pts:
        raise ValueError("Empty trajectory line.")
    return pts

def parse_trajectories(path: str, A: int):
    with open(path, "r", encoding="utf-8") as fh:
        raw = [ln.strip() for ln in fh if ln.strip() and not ln.lstrip().startswith("#")]
    if not raw:
        raise ValueError("Empty trajectory file.")
    lines = [parse_traj_line(s) for s in raw]
    if len(lines) % A != 0:
        raise ValueError("Trajectory lines must be a multiple of agents (A).")
    iters = [lines[i:i + A] for i in range(0, len(lines), A)]
    return iters

def center(x: float, y: float) -> Tuple[float, float]:
    return (x + 0.5, y + 0.5)

def heading_degrees(prev_xy: Tuple[float, float], next_xy: Tuple[float, float]) -> float:
    dx = next_xy[0] - prev_xy[0]
    dy = next_xy[1] - prev_xy[1]
    if dx == 0 and dy == 0:
        return 0.0
    return np.degrees(np.arctan2(dy, dx))

def mouth_angle_degrees(t: int, base: float = 22.0, amp: float = 16.0, period: int = 4) -> float:
    return float(base + amp * 0.5 * (1.0 - np.cos(2.0 * np.pi * (t % period) / period)))

def hex_to_rgb(h: str):
    h = h.lstrip('#')
    return tuple(int(h[i:i+2], 16)/255.0 for i in (0, 2, 4))

def rgb_to_hex(rgb):
    r = int(max(0, min(255, round(rgb[0]*255))))
    g = int(max(0, min(255, round(rgb[1]*255))))
    b = int(max(0, min(255, round(rgb[2]*255))))
    return f"#{r:02X}{g:02X}{b:02X}"

def lighten_color(rgb, factor=0.2):
    r, g, b = rgb
    return (r + (1 - r) * factor, g + (1 - g) * factor, b + (1 - b) * factor)

def lighten_hex(c: str, factor=0.2) -> str:
    return rgb_to_hex(lighten_color(hex_to_rgb(c), factor))

# ---------- colors ----------
import colorsys

def generate_distinct_colors(n: int, seed: int | None = None):
    """
    Randomly sample distinct HSV tuples with varied saturation/value so colors
    aren't just different hues. Ensures no duplicates.
    If seed is None -> different palette each run; set a seed for reproducibility.
    """
    rng = random.Random(seed)
    # good ranges: avoid very dark or washed-out
    S_MIN, S_MAX = 0.55, 0.95
    V_MIN, V_MAX = 0.80, 0.98

    colors = []
    seen = set()
    # Use low-discrepancy hues (golden ratio) + random jitter for spacing
    phi = 0.6180339887498949
    base = rng.random()
    for i in range(n * 2):  # oversample to be safe
        if len(colors) >= n:
            break
        h = (base + i * phi + rng.uniform(-0.03, 0.03)) % 1.0
        s = rng.uniform(S_MIN, S_MAX)
        v = rng.uniform(V_MIN, V_MAX)
        key = (round(h, 4), round(s, 3), round(v, 3))
        if key in seen:
            continue
        seen.add(key)
        rgb = colorsys.hsv_to_rgb(h, s, v)
        colors.append(rgb_to_hex(rgb))
    # If somehow not enough (extremely unlikely), fill remaining purely random
    while len(colors) < n:
        h = rng.random()
        s = rng.uniform(S_MIN, S_MAX)
        v = rng.uniform(V_MIN, V_MAX)
        key = (round(h, 4), round(s, 3), round(v, 3))
        if key in seen:
            continue
        seen.add(key)
        colors.append(rgb_to_hex(colorsys.hsv_to_rgb(h, s, v)))
    return colors

def generate_colors(n: int):
    """Generate up to n distinct colors using HSV spectrum."""
    colors = []
    for i in range(n):
        hue = i / n
        rgb = colorsys.hsv_to_rgb(hue, 0.8, 0.95)
        colors.append(rgb_to_hex(rgb))
    return colors

# Create up to 2000 distinct colors
PAC_COLORS = generate_colors(1000)

# ---------- main ----------

def main():
    ap = argparse.ArgumentParser(
        description="Pac-Man agents on the last iteration; time-... has food; agents at cell centers; t starts at 0; stops at end."
    )
    ap.add_argument("map", help=".map file (MovingAI format)")
    ap.add_argument("scen", help=".scen file (MovingAI format)")
    ap.add_argument("traj", help="t1.sol (one line per agent per iteration; semicolon-separated x,y)")
    ap.add_argument("-A", "--agents", type=int, required=True, help="Number of agents to use from .scen / trajectories")
    ap.add_argument("--interval", type=int, default=600, help="ms per time step")
    ap.add_argument("--save", help="Optional output (mp4/gif)")
    args = ap.parse_args()

    rows, cols, obstacles, A, starts, goals = parse_instance(args.map, args.scen, args.agents)

    # --- Size scaling with number of agents ---
    def scale_value(base, A, A_min=10, A_max=128, min_scale=0.5):
        """
        Linearly scale 'base' size between 1.0× (A <= A_min)
        and min_scale× (A >= A_max).
        """
        if A <= A_min:
            return base
        elif A >= A_max:
            return base * min_scale
        else:
            f = 1.0 - (A - A_min) / (A_max - A_min)
            return base * (min_scale + f * (1 - min_scale))

    # Base sizes
    BASE_PAC_RADIUS = 0.38
    BASE_GOAL_RING  = 0.24
    BASE_GOAL_DOT   = 0.12

    # Scaled sizes for current number of agents
    PACMAN_RADIUS = scale_value(BASE_PAC_RADIUS, A)
    GOAL_RING_RADIUS = scale_value(BASE_GOAL_RING, A)
    GOAL_DOT_RADIUS  = scale_value(BASE_GOAL_DOT, A)


    # ***IMPORTANT: use LAST iteration exactly like original***
    iters = parse_trajectories(args.traj, A)
    last_iter = iters[-1]

    # Centered trajectories from last iteration
    trajs = [[center(x, y) for (x, y) in last_iter[a]] for a in range(A)]
    lengths = [len(tr) for tr in trajs]
    T_max = max(lengths) if lengths else 1

    # Figure + axes
    fig, ax = plt.subplots(figsize=(max(6, cols*0.45), max(5, rows*0.45)))
    ax.set_aspect('equal')
    ax.set_xlim(0, cols)
    ax.set_ylim(0, rows)
    # ax.set_title("Pac-Man MAPF — last iteration")
    ax.set_xticks([]) 
    ax.set_yticks([]) 
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    # Grid
    for x in range(cols + 1):
        ax.add_line(plt.Line2D([x, x], [0, rows], lw=0.3, color="lightgray", zorder=0))
    for y in range(rows + 1):
        ax.add_line(plt.Line2D([0, cols], [y, y], lw=0.3, color="lightgray", zorder=0))

    # Obstacles
    for (ox, oy) in obstacles:
        ax.add_patch(Rectangle((ox, oy), 1, 1, facecolor="#555555", edgecolor="#444444",
                               linewidth=0.5, zorder=0))

    # Goals + pacmen
    goal_dots_inner, goal_dots_ring = [], []
    pac_patches: List[Wedge] = []

    PAC_COLORS = generate_distinct_colors(A, seed=0)

    for a in range(A):
        color = PAC_COLORS[a]
        food_color = lighten_hex(color, 0.25)
        gx, gy = goals[a]
        cx, cy = center(gx, gy)

        ring = Circle((cx, cy), radius=GOAL_RING_RADIUS, facecolor="none", edgecolor=color, linewidth=1.6, zorder=2)
        dot  = Circle((cx, cy), radius=GOAL_DOT_RADIUS, facecolor=food_color, edgecolor="black", linewidth=0.3, zorder=3)

        ax.add_patch(ring); ax.add_patch(dot)
        goal_dots_ring.append(ring); goal_dots_inner.append(dot)

        pac = Wedge((0.5, 0.5), r=PACMAN_RADIUS, theta1=22.0, theta2=-22.0, facecolor=color, edgecolor="black", linewidth=0.6, zorder=5)

        ax.add_patch(pac)
        pac_patches.append(pac)

    # HUD above top-right corner of the map
    fig.canvas.draw()
    axpos = ax.get_position()
    hud_x = axpos.x1 - 0.07   # slightly to the left of the right edge
    hud_y = axpos.y1 + 0.07  # above the top of the axes
    hud = fig.text(hud_x, hud_y, "t = 0",
               va="bottom", ha="right", fontsize=12,
               bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85, edgecolor="#ccc"))

    last_headings = [0.0] * A

    # PREPOSITION at t=0
    for a in range(A):
        traj = trajs[a]
        if len(traj) == 0:
            continue
        cx, cy = traj[0]
        pac_patches[a].set_center((cx, cy))
        if len(traj) >= 2:
            head = heading_degrees(traj[0], traj[1])
            last_headings[a] = head
            pac_patches[a].set_theta1(head + mouth_angle_degrees(0))
            pac_patches[a].set_theta2(head - mouth_angle_degrees(0))

    # ---------- INTERACTIVE mode (no --save): show buttons/slider ----------
    if not args.save:
        # Leave bottom margin for widgets
        plt.subplots_adjust(left=0.08, right=0.98, top=0.93, bottom=0.30)
        fig.canvas.draw()
        axpos = ax.get_position()

        # Slider
        slider_h = 0.045
        slider_gap = 0.11
        slider_y = axpos.y0 - slider_gap
        ax_slide = plt.axes([axpos.x0, slider_y, axpos.width, slider_h])
        slider = Slider(ax_slide, "Time", 0, max(0, T_max - 1), valinit=0, valstep=1)

        # Buttons row
        btn_h = 0.06
        btn_w = 0.12
        btn_gap = 0.03
        group_w = 4*btn_w + 3*btn_gap
        group_left = axpos.x0 + (axpos.width - group_w)/2.0
        btn_y = slider_y - (btn_h + 0.02)
        ax_prev   = plt.axes([group_left + 0*(btn_w + btn_gap), btn_y, btn_w, btn_h])
        ax_play   = plt.axes([group_left + 1*(btn_w + btn_gap), btn_y, btn_w, btn_h])
        ax_next   = plt.axes([group_left + 2*(btn_w + btn_gap), btn_y, btn_w, btn_h])
        ax_restart= plt.axes([group_left + 3*(btn_w + btn_gap), btn_y, btn_w, btn_h])

        b_prev    = Button(ax_prev,   "Prev")
        b_play    = Button(ax_play,   "Pause")
        b_next    = Button(ax_next,   "Next")
        b_restart = Button(ax_restart,"Restart")

        t_cur = 0
        playing = True
        def set_button_playing(is_playing: bool):
            b_play.label.set_text("Pause" if is_playing else "Play")
        set_button_playing(playing)

        def update_widgets(t_cur):
            # Update pacmen and HUD for time t_cur
            for a in range(A):
                traj = trajs[a]
                L = lengths[a]
                if L == 0: continue
                tA = min(t_cur, L - 1)
                cx, cy = traj[tA]
                if tA < L - 1:
                    head = heading_degrees(traj[tA], traj[tA + 1])
                    if head != 0.0: last_headings[a] = head
                elif tA >= 1:
                    head = heading_degrees(traj[tA - 1], traj[tA])
                    if head != 0.0: last_headings[a] = head
                else:
                    head = last_headings[a]
                mdeg = mouth_angle_degrees(t_cur, base=22.0, amp=16.0, period=4)
                pac = pac_patches[a]
                pac.set_center((cx, cy))
                pac.set_theta1(head + mdeg)
                pac.set_theta2(head - mdeg)
            hud.set_text(f"t = {t_cur}")
            fig.canvas.draw_idle()

        def init():
            return pac_patches + goal_dots_inner + goal_dots_ring

        def update(_frame):
            nonlocal t_cur, playing
            if playing:
                if t_cur < T_max - 1:
                    t_cur += 1
                    slider.set_val(t_cur)  # triggers on_slider
                else:
                    playing = False
                    set_button_playing(False)
            return pac_patches + goal_dots_inner + goal_dots_ring

        def on_slider(val):
            nonlocal t_cur
            t_cur = int(val)
            update_widgets(t_cur)

        def do_prev(_):
            nonlocal t_cur
            if t_cur > 0:
                t_cur -= 1
                slider.set_val(t_cur)

        def do_next(_):
            nonlocal t_cur
            if t_cur < T_max - 1:
                t_cur += 1
                slider.set_val(t_cur)

        def do_play(_):
            nonlocal playing
            playing = not playing
            set_button_playing(playing)

        def do_restart(_):
            nonlocal t_cur, playing
            t_cur = 0
            slider.set_val(0)
            playing = True
            set_button_playing(True)

        slider.on_changed(on_slider)
        b_prev.on_clicked(do_prev)
        b_next.on_clicked(do_next)
        b_play.on_clicked(do_play)
        b_restart.on_clicked(do_restart)

        anim = FuncAnimation(
            fig, update, frames=None,
            init_func=init, interval=args.interval, blit=False, cache_frame_data=False
        )
        plt.show()

    # ---------- SAVE mode (with --save): no widgets, deterministic ----------
    else:
        # Make sure there's enough top margin for the HUD outside axes when saving
        plt.subplots_adjust(top=0.93, bottom=0.06, left=0.08, right=0.98)
        fig.canvas.draw_idle()

        def render_t(t):
            for a in range(A):
                traj = trajs[a]
                L = lengths[a]
                if L == 0: continue
                tA = min(t, L - 1)
                cx, cy = traj[tA]
                if tA < L - 1:
                    head = heading_degrees(traj[tA], traj[tA + 1])
                    if head != 0.0: last_headings[a] = head
                elif tA >= 1:
                    head = heading_degrees(traj[tA - 1], traj[tA])
                    if head != 0.0: last_headings[a] = head
                else:
                    head = last_headings[a]
                mdeg = mouth_angle_degrees(t, base=22.0, amp=16.0, period=4)
                pac = pac_patches[a]
                pac.set_center((cx, cy))
                pac.set_theta1(head + mdeg)
                pac.set_theta2(head - mdeg)
            hud.set_text(f"t = {t}")
            return pac_patches + goal_dots_inner + goal_dots_ring

        save_anim = FuncAnimation(
            fig, lambda f: render_t(f), frames=range(T_max),
            interval=args.interval, blit=False, repeat=False
        )
        ext = os.path.splitext(args.save)[1].lower()
        if ext == ".mp4":
            save_anim.save(args.save, dpi=150, fps=max(1, int(1000/args.interval)))
        else:
            save_anim.save(args.save, dpi=150, writer="pillow", fps=max(1, int(1000/args.interval)))

if __name__ == "__main__":
    main()

