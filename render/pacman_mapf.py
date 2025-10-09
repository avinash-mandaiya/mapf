# pacman_mapf.py
# Usage:
#   python pacman_mapf.py instance2.txt t1.sol
#   # optional:
#   # python pacman_mapf.py instance2.txt t1.sol --interval 600
#   # MPLBACKEND=Agg python pacman_mapf.py instance2.txt t1.sol --save out.mp4

import argparse
from typing import List, Tuple, Optional
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button, Slider
from matplotlib.patches import Rectangle, Wedge, Circle
import os

# ---------- parsing: instance/setup ----------

def _next_nonempty_tokens(lines_iter) -> Optional[List[str]]:
    for raw in lines_iter:
        s = raw.strip()
        if not s or s.lstrip().startswith("#"):
            continue
        return s.split()
    return None

def parse_instance(path: str):
    """
    instance2.txt:

      rows cols
      Zobs
      Zobs lines: x y
      Nagents
      Nagents lines: sx sy gx gy
    """
    path = Path(path)
    if not path.is_absolute():
        path = Path(__file__).resolve().parent.parent / path

    with open(path, "r", encoding="utf-8") as fh:
        lines = fh.readlines()
    it = iter(lines)

    toks = _next_nonempty_tokens(it)
    if toks is None or len(toks) < 2:
        raise ValueError("Missing 'rows cols'.")
    rows, cols = int(toks[1]), int(toks[0])

    toks = _next_nonempty_tokens(it)
    if toks is None:
        raise ValueError("Missing obstacle count.")
    zobs = int(toks[0])

    obstacles = []
    for _ in range(zobs):
        toks = _next_nonempty_tokens(it)
        if toks is None or len(toks) < 2:
            raise ValueError("Incomplete obstacle coords.")
        obstacles.append((int(toks[0]), int(toks[1])))

    toks = _next_nonempty_tokens(it)
    if toks is None:
        raise ValueError("Missing agents count.")
    A = int(toks[0])
    if A < 1:
        raise ValueError("Agents must be >= 1")

    starts, goals = [], []
    for _ in range(A):
        toks = _next_nonempty_tokens(it)
        if toks is None or len(toks) < 4:
            raise ValueError("Each agent needs: sx sy gx gy")
        sx, sy, gx, gy = map(int, toks[:4])
        starts.append((sx, sy))
        goals.append((gx, gy))

    return rows, cols, obstacles, A, starts, goals

# ---------- parsing: trajectories (t1.sol) ----------

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
    """
    t1.sol: one line per agent per iteration (semicolon-separated x,y).
    Total lines must be a multiple of A.
    """
    with open(path, "r", encoding="utf-8") as fh:
        raw = [ln.strip() for ln in fh if ln.strip() and not ln.lstrip().startswith("#")]
    if not raw:
        raise ValueError("Empty trajectory file.")
    lines = [parse_traj_line(s) for s in raw]
    if len(lines) % A != 0:
        raise ValueError("Trajectory lines must be a multiple of agents (A).")
    iters = [lines[i:i + A] for i in range(0, len(lines), A)]
    return iters

# ---------- helpers ----------

def center(x: float, y: float) -> Tuple[float, float]:
    """Place at cell center."""
    return (x + 0.5, y + 0.5)

def heading_degrees(prev_xy: Tuple[float, float], next_xy: Tuple[float, float]) -> float:
    dx = next_xy[0] - prev_xy[0]
    dy = next_xy[1] - prev_xy[1]
    if dx == 0 and dy == 0:
        return 0.0
    return np.degrees(np.arctan2(dy, dx))  # 0° = +x axis

def mouth_angle_degrees(t: int, base: float = 22.0, amp: float = 16.0, period: int = 4) -> float:
    # Chewing over time step t (0..)
    return float(base + amp * 0.5 * (1.0 - np.cos(2.0 * np.pi * (t % period) / period)))

def hex_to_rgb(h: str):
    h = h.lstrip('#')
    return tuple(int(h[i:i+2], 16)/255.0 for i in (0, 2, 4))

def rgb_to_hex(rgb):
    return '#%02x%02x%02x' % tuple(max(0, min(255, int(round(c*255)))) for c in rgb)

def lighten(hex_color: str, factor: float = 0.5) -> str:
    """Mix color with white by 'factor' (0..1)."""
    r, g, b = hex_to_rgb(hex_color)
    r = r + (1 - r)*factor
    g = g + (1 - g)*factor
    b = b + (1 - b)*factor
    return rgb_to_hex((r, g, b))

PAC_COLORS = [
    "#F7D038", "#2ca02c", "#d62728", "#9467bd",
    "#ff7f0e", "#17becf", "#e377c2", "#8c564b",
    "#7f7f7f", "#bcbd22"
]

# ---------- main ----------

def main():
    ap = argparse.ArgumentParser(
        description="Pac-Man agents on the last iteration; time-based animation; only goal has food; agents at cell centers; t starts at 0; stops at end."
    )
    ap.add_argument("instance", help="instance2.txt (rows cols; obstacles; Nagents; sx sy gx gy)")
    ap.add_argument("traj", help="t1.sol (one line per agent per iteration; semicolon-separated x,y)")
    ap.add_argument("--interval", type=int, default=600, help="ms per time step")
    ap.add_argument("--save", help="Optional output (mp4/gif)")
    args = ap.parse_args()

    rows, cols, obstacles, A, starts, goals = parse_instance(args.instance)
    iters = parse_trajectories(args.traj, A)
    last_iter = iters[-1]

    # Centered trajectories from last iteration
    trajs = [[center(x, y) for (x, y) in last_iter[a]] for a in range(A)]
    lengths = [len(tr) for tr in trajs]
    T_max = max(lengths) if lengths else 1  # time steps (indices 0..T_max-1)

    # First index where each agent reaches its goal cell; default to last index
    goal_indices = []
    for a in range(A):
        gx_c, gy_c = center(*goals[a])
        idx = None
        for i, (px, py) in enumerate(trajs[a]):
            if abs(px - gx_c) < 1e-9 and abs(py - gy_c) < 1e-9:
                idx = i
                break
        if idx is None:
            idx = max(0, lengths[a] - 1)
        goal_indices.append(idx)

    # Figure/axes
    fig, ax = plt.subplots()
    # room for full-width time slider + centered buttons
    plt.subplots_adjust(left=0.08, right=0.98, top=0.93, bottom=0.30)

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim((0, cols)); ax.set_ylim((0, rows))
    ax.set_xticks(range(0, cols + 1))
    ax.set_yticks(range(0, rows + 1))
    ax.grid(True)
    ax.set_title("")

    # HUD outside axes — just the time index
    hud = fig.text(0.985, 0.965, "", ha="right", va="top", fontsize=10,
                   bbox=dict(boxstyle="round", facecolor="white", alpha=0.7, edgecolor="#ccc"))

    # Obstacles
    for (ox, oy) in obstacles:
        ax.add_patch(Rectangle((ox, oy), 1, 1, facecolor="#555555", edgecolor="#444444",
                               linewidth=0.5, zorder=0))

    # Goal markers (same color as agent) + one goal pellet (only at goal)
    goal_dots_inner = []
    goal_dots_ring = []
    goal_pellets = []

    pellet_radius = 0.12
    pac_patches: List[Wedge] = []

    for a in range(A):
        color = PAC_COLORS[a % len(PAC_COLORS)]
        food_color = lighten(color, 0.55)

        gx, gy = center(*goals[a])

        # goal marker (inner + ring)
        gd_i, = ax.plot([gx], [gy], marker="o", markersize=6, color=color,
                         markeredgecolor=color, markeredgewidth=0.8, linestyle="None", zorder=4)
        gd_r, = ax.plot([gx], [gy], marker="o", markersize=9, fillstyle="none",
                         markeredgecolor="black", markeredgewidth=1.2, linestyle="None", zorder=4)
        goal_dots_inner.append(gd_i)
        goal_dots_ring.append(gd_r)

        # GOAL FOOD (only at goal)
        pellet = Circle((gx, gy), pellet_radius, facecolor=food_color,
                        edgecolor="black", linewidth=0.6, zorder=2)
        ax.add_patch(pellet)
        goal_pellets.append(pellet)

        # Pac-Man wedge
        start_center = trajs[a][0] if lengths[a] > 0 else center(0, 0)
        pac = Wedge(start_center, 0.32, 0, 360,
                    facecolor=color, edgecolor="black", linewidth=1.0, zorder=5)
        ax.add_patch(pac)
        pac_patches.append(pac)

    # UI: Time slider (full width) + centered buttons
    fig.canvas.draw()
    axpos = ax.get_position()
    slider_h = 0.045
    slider_gap = 0.11
    slider_y = axpos.y0 - slider_gap
    ax_slide = plt.axes([axpos.x0, slider_y, axpos.width, slider_h])
    slider = Slider(ax_slide, "Time", 0, max(0, T_max - 1), valinit=0, valstep=1)

    btn_h = 0.06
    btn_w = 0.12
    btn_gap = 0.03
    # 4 buttons now: Prev | Pause/Play | Next | Restart
    group_w = 4*btn_w + 3*btn_gap
    group_left = axpos.x0 + (axpos.width - group_w)/2.0
    btn_y = slider_y - (btn_h + 0.025)

    ax_prev    = plt.axes([group_left + 0*(btn_w+btn_gap), btn_y, btn_w, btn_h])
    ax_play    = plt.axes([group_left + 1*(btn_w+btn_gap), btn_y, btn_w, btn_h])
    ax_next    = plt.axes([group_left + 2*(btn_w+btn_gap), btn_y, btn_w, btn_h])
    ax_restart = plt.axes([group_left + 3*(btn_w+btn_gap), btn_y, btn_w, btn_h])

    btn_prev    = Button(ax_prev, "Prev")
    btn_play    = Button(ax_play, "Pause")
    btn_next    = Button(ax_next, "Next")
    btn_restart = Button(ax_restart, "Restart")
    
    ui_axes = [ax_slide, ax_prev, ax_play, ax_next]
    try:
        ui_axes.append(ax_restart)  # only if you added Restart
    except NameError:
        pass

    # Animation state
    t_cur = 0
    paused = False
    last_headings = [0.0]*A

    def draw_time(t_index: int):
        nonlocal t_cur
        t_cur = min(max(0, t_index), T_max - 1)

        # hide pellet once agent reaches its goal index
        for a in range(A):
            goal_pellets[a].set_visible(t_cur < goal_indices[a])

        # update Pac-Men
        for a in range(A):
            traj = trajs[a]
            L = lengths[a]
            if L == 0:
                continue
            tA = min(t_cur, L - 1)
            cx, cy = traj[tA]

            # heading
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
        draw_time(0)

    # freeze-at-end update (no timer stop)
    def update(_):
        nonlocal t_cur, paused
        if paused:
            return
        if t_cur < T_max - 1:
            t_next = t_cur + 1
            slider.eventson = False
            slider.set_val(t_next)
            slider.eventson = True
            draw_time(t_next)
        else:
            # reached last step → freeze; user can press Restart or Prev/Play
            paused = True
            btn_play.label.set_text("Play")

    # controls
    def do_prev(event=None):
        nonlocal paused, t_cur
        paused = True
        btn_play.label.set_text("Play")
        draw_time(t_cur - 1)

    def do_next(event=None):
        nonlocal paused, t_cur
        paused = True
        btn_play.label.set_text("Play")
        draw_time(t_cur + 1)

    def do_play(event=None):
        nonlocal paused, t_cur
        # auto-rewind if at the end
        if paused and t_cur >= T_max - 1 and T_max > 0:
            slider.eventson = False
            slider.set_val(0)
            slider.eventson = True
            draw_time(0)
        paused = not paused
        btn_play.label.set_text("Pause" if not paused else "Play")

    def do_restart(event=None):
        nonlocal paused, t_cur
        # rewind to t=0 and pause (so user can hit Play)
        t_cur = 0
        slider.eventson = False
        slider.set_val(0)
        slider.eventson = True
        draw_time(0)
        paused = True
        btn_play.label.set_text("Play")

    btn_prev.on_clicked(do_prev)
    btn_next.on_clicked(do_next)
    btn_play.on_clicked(do_play)
    btn_restart.on_clicked(do_restart)
    slider.on_changed(on_slide := (lambda val: (do_play("pause") if False else None,  # keep API stable; not used
                                                (paused := True, btn_play.label.set_text("Play"), draw_time(int(val))))))

    def on_key(event):
        if event.key in (" ", "space"):
            do_play()
        elif event.key in ("right", "n", "j"):
            do_next()
        elif event.key in ("left", "p", "k"):
            do_prev()

    fig.canvas.mpl_connect("key_press_event", on_key)

    # ---------- Saving (finite frames) vs Interactive (infinite ticks) ----------

    if args.save:
        fps = max(1, 1000 // args.interval)
        
        ax.tick_params(axis='both', which='both',
               bottom=False, top=False, left=False, right=False,
               labelbottom=False, labelleft=False)
    
        # --- HIDE UI FOR SAVING ---
        for a in ui_axes:
            a.set_visible(False)
        # tighter bottom margin so the plot fills the video (keep the HUD at top)
        plt.subplots_adjust(left=0.08, right=0.98, top=0.93, bottom=0.08)
        fig.canvas.draw_idle()
    
        def update_save(f):
            draw_time(int(f))   # render exact time index
            return []
    
        save_anim = FuncAnimation(
            fig, update_save, frames=range(T_max),
            init_func=lambda: draw_time(0), repeat=False, blit=False,
            cache_frame_data=False
        )
    
        ext = args.save.lower().split(".")[-1]
        if ext == "mp4":
            save_anim.save(args.save, writer="ffmpeg", fps=fps)
        elif ext == "gif":
            save_anim.save(args.save, writer="imagemagick", fps=fps)
        else:
            print("Unknown extension; use .mp4 or .gif.")
    
        # --- RESTORE UI FOR INTERACTIVE USE ---
        # (skip this if you're running headless with Agg)
        if not plt.get_backend().lower().startswith("agg"):
            for a in ui_axes:
                a.set_visible(True)
            # restore the layout you use for interactive UI
            plt.subplots_adjust(left=0.08, right=0.98, top=0.93, bottom=0.30)
            fig.canvas.draw_idle()
    

    # Interactive animation (infinite tick source; we freeze at end so it doesn’t loop)
    anim = FuncAnimation(
        fig, update, frames=None,
        init_func=init, interval=args.interval, blit=False, cache_frame_data=False
    )
    plt.show()

if __name__ == "__main__":
    main()

