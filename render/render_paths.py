# render_paths.py
# Usage:
#   python render_paths.py ../benchmarks/maze-32-32-2.map ../benchmarks/scen-even/maze-32-32-2-even-1.scen t3.sol 2 --save pac1.mp4 --interval 400
#   # optional:
#   # MPLBACKEND=Agg python render_iters.py instance2.txt t1.sol --save out.mp4

import argparse
from typing import List, Tuple, Optional
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from matplotlib.widgets import Button, Slider
from matplotlib.patches import Rectangle
from pathlib import Path

# ---------- Parsing: instance/setup ----------

def _next_nonempty_tokens(lines_iter) -> Optional[List[str]]:
    for raw in lines_iter:
        s = raw.strip()
        if not s or s.lstrip().startswith("#"):
            continue
        return s.split()
    return None

def load_movingai_map(map_path: str):
    """
    Returns: (width, height, free[x][y]) with origin at bottom-left.
    free[x][y] = 1 if traversable ('.' or 'G'), else 0.
    """
    p = Path(map_path)
    lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
    if not lines or not lines[0].startswith("type"):
        raise ValueError("Invalid map header: expected 'type ...' on first line.")

    try:
        H = int(lines[1].split()[1])
        W = int(lines[2].split()[1])
    except Exception as e:
        raise ValueError("Failed to parse 'height'/'width' lines in the map.") from e

    if lines[3].strip() != "map":
        raise ValueError("Expected a line 'map' after width/height.")

    rows = lines[4:4+H]
    if len(rows) != H:
        raise ValueError("Map file truncated (not enough rows).")

    def is_free(ch: str) -> bool:
        return ch in ('.', 'G')

    # column-major free[x][y], math coords (y=0 bottom)
    free = [[0 for _ in range(H)] for _ in range(W)]
    for y_file, row in enumerate(rows):
        if len(row) != W:
            raise ValueError(f"Row width mismatch at file row {y_file}: got {len(row)}, expected {W}")
        y = H - 1 - y_file  # flip to bottom-left origin
        for x in range(W):
            free[x][y] = 1 if is_free(row[x]) else 0
    return W, H, free


def load_movingai_scen(scen_path: str, k: int, map_height: int):
    """
    Load FIRST k agents from .scen and convert to math coords (bottom-left origin).
    Returns: list of (sx, sy, gx, gy) in math coords.
    """
    p = Path(scen_path)
    with p.open("r", encoding="utf-8", errors="replace") as fh:
        header = fh.readline()
        if not header or not header.startswith("version"):
            raise ValueError("Scenario file missing 'version' header.")

        agents = []
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 9:
                raise ValueError(f"Bad .scen line: {line}")
            # fields: bucket map_path map_w map_h sx sy gx gy optimal_len
            _, _, mw, mh, sx_f, sy_f, gx_f, gy_f, _ = parts[:9]
            sx_f, sy_f, gx_f, gy_f = map(int, (sx_f, sy_f, gx_f, gy_f))
            # flip Y from file coords (top-left) to math coords (bottom-left)
            sx, sy = sx_f, (map_height - 1 - sy_f)
            gx, gy = gx_f, (map_height - 1 - gy_f)
            agents.append((sx, sy, gx, gy))
            if k > 0 and len(agents) >= k:
                break
    if k > 0 and len(agents) < k:
        raise ValueError(f"Scenario has fewer than k={k} usable lines.")
    return agents



def parse_instance(path: str):
    """
    instance2.txt format:

      rows cols
      Zobs
      Zobs lines of: x y
      Nagents
      Nagents lines of: sx sy gx gy
    """
    path = Path(path)
    if not path.is_absolute():
        path = Path(__file__).resolve().parent.parent / path

    with open(path, "r", encoding="utf-8") as fh:
        lines = fh.readlines()

    it = iter(lines)

    # Grid size
    toks = _next_nonempty_tokens(it)
    if toks is None or len(toks) < 2:
        raise ValueError("Missing 'rows cols' line.")
    rows, cols = int(toks[1]), int(toks[0])

    # Obstacles count
    toks = _next_nonempty_tokens(it)
    if toks is None or len(toks) < 1:
        raise ValueError("Missing obstacle count.")
    zobs = int(toks[0])

    # Obstacles list
    obstacles = []
    for _ in range(zobs):
        toks = _next_nonempty_tokens(it)
        if toks is None or len(toks) < 2:
            raise ValueError("Incomplete obstacle coordinates.")
        x, y = int(toks[0]), int(toks[1])
        obstacles.append((x, y))

    # Agents count
    toks = _next_nonempty_tokens(it)
    if toks is None or len(toks) < 1:
        raise ValueError("Missing agents count.")
    A = int(toks[0])
    if A < 1:
        raise ValueError("Agents must be >= 1")

    # Agent start/goal pairs
    starts, goals = [], []
    for _ in range(A):
        toks = _next_nonempty_tokens(it)
        if toks is None or len(toks) < 4:
            raise ValueError("Each agent needs: sx sy gx gy")
        sx, sy, gx, gy = map(int, toks[:4])
        starts.append((sx, sy))
        goals.append((gx, gy))

    return rows, cols, obstacles, A, starts, goals

# ---------- Parsing: trajectories (t1.sol) ----------

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
    t1.sol format:
      One line per agent per iteration (semicolon-separated x,y positions).
      Total number of lines must be a multiple of A.
    """
    with open(path, "r", encoding="utf-8") as fh:
        raw_lines = [ln.strip() for ln in fh if ln.strip() and not ln.lstrip().startswith("#")]

    if not raw_lines:
        raise ValueError("Empty trajectory file.")

    lines_parsed = [parse_traj_line(s) for s in raw_lines]
    if len(lines_parsed) % A != 0:
        raise ValueError("Trajectory lines must be a multiple of agents (A).")

    # Group into iterations (no need for equal time-length across iterations)
    iters = [lines_parsed[i:i + A] for i in range(0, len(lines_parsed), A)]
    return iters

# ---------- Geometry / Offsets ----------

def agent_offset(a0: int, A: int) -> Tuple[float, float]:
    """
    Offsets per your rule:
      A = 1: (0.5, 0.5)
      A > 1: (0.25,0.25) + ((i-1)/(A-1)) * (0.5,0.5), where i = a0+1
    """
    if A <= 1:
        return (0.5, 0.5)
    i = a0 + 1
    alpha = (i - 1) / (A - 1)
    dx = 0.25 + 0.5 * alpha
    dy = 0.25 + 0.5 * alpha
    return (dx, dy)

def offset_traj(traj: List[Tuple[float, float]], a0: int, A: int):
    dx, dy = agent_offset(a0, A)
    return [(x + dx, y + dy) for (x, y) in traj]

# ---------- Main ----------

def main():
    ap = argparse.ArgumentParser(
        description="Per-iteration renderer: grid & obstacles from instance file; paths from trajectory file; agent offsets and darker-start blue gradient."
    )

    ap.add_argument("map",  help=".map file (MovingAI ASCII grid)")
    ap.add_argument("scen", help=".scen file (MovingAI scenario list)")
    ap.add_argument("traj", help="trajectories file")
    ap.add_argument("-A", "--agents", type=int, required=True, help="number of agents to load from .scen (first A lines)")
    ap.add_argument("--interval", type=int, default=800, help="ms between frames (iterations)")
    ap.add_argument("--save", help="Optional output (mp4/gif)")
    args = ap.parse_args()

    # Load map (W,H,free) and first A agents from scen
    W, H, free_xy = load_movingai_map(args.map)
    agents = load_movingai_scen(args.scen, args.agents, H)

    iters = parse_trajectories(args.traj, A)
    nIters = len(iters)

    # Plot bounds from rows/cols (full cells)
    xr = (0, cols)
    yr = (0, rows)

    fig, ax = plt.subplots()
    # generous bottom margin for slider + centered buttons
    plt.subplots_adjust(left=0.08, right=0.98, top=0.93, bottom=0.30)

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(xr); ax.set_ylim(yr)
    ax.set_xticks(range(0, cols + 1))
    ax.set_yticks(range(0, rows + 1))
    ax.grid(True, which="both")
    ax.set_title("")  # keep title empty

    # HUD outside axes (top-right of figure)
    hud = fig.text(0.985, 0.965, "", ha="right", va="top", fontsize=10,
                   bbox=dict(boxstyle="round", facecolor="white", alpha=0.7, edgecolor="#ccc"))

    # Obstacles as dark grey squares (behind paths)
    for (ox, oy) in obstacles:
        ax.add_patch(Rectangle((ox, oy), 1, 1, facecolor="#555555", edgecolor="#444444",
                               linewidth=0.5, zorder=0))

    # Gradient paths (blue; darker at start)
    norm = Normalize(vmin=0.0, vmax=1.0)
    lcs = [LineCollection([], cmap="Blues", norm=norm, linewidths=2, zorder=2) for _ in range(A)]
    for lc in lcs:
        ax.add_collection(lc)

    # Start & Goal markers (same color; goal has black ring)
    pts_start = [ax.plot([], [], marker="o", markersize=6, color="#1f77b4",
                         markeredgecolor="black", markeredgewidth=0.8,
                         linestyle="None", zorder=3)[0] for _ in range(A)]
    pts_goal_inner = [ax.plot([], [], marker="o", markersize=6, color="#1f77b4",
                              markeredgecolor="#1f77b4", markeredgewidth=0.8,
                              linestyle="None", zorder=4)[0] for _ in range(A)]
    pts_goal_ring = [ax.plot([], [], marker="o", markersize=9, fillstyle="none",
                             markeredgecolor="black", markeredgewidth=1.2,
                             linestyle="None", zorder=4)[0] for _ in range(A)]

    current_iter = 0
    paused = False

    def draw_iteration(i):
        it = iters[i]  # list of A trajectories (each a list of (x,y))
        for a in range(A):
            # Path (whole thing at once)
            traj = offset_traj(it[a], a, A)
            pts = np.asarray(traj, dtype=float)
            if len(pts) >= 2:
                segs = np.stack([pts[:-1], pts[1:]], axis=1)
                vals = np.linspace(0.4, 1.0, len(segs))  # darker start -> dark end
                lcs[a].set_segments(segs)
                lcs[a].set_array(vals)
            else:
                lcs[a].set_segments([])
                lcs[a].set_array(None)

            # Start marker
            sx, sy = starts[a]
            offx, offy = agent_offset(a, A)
            pts_start[a].set_data([sx + offx], [sy + offy])

            # Goal marker (inner + ring)
            gx, gy = goals[a]
            pts_goal_inner[a].set_data([gx + offx], [gy + offy])
            pts_goal_ring[a].set_data([gx + offx], [gy + offy])

        hud.set_text(f"iter: {i + 1} / {nIters}")
        fig.canvas.draw_idle()

    def init():
        draw_iteration(current_iter)

    # Animation callback: one frame per iteration
    def update(frame):
        nonlocal current_iter
        if not paused:
            current_iter = frame % nIters
            slider.eventson = False
            slider.set_val(current_iter)
            slider.eventson = True
            draw_iteration(current_iter)

    # --- UI: slider same width as axes; centered buttons beneath ---
    fig.canvas.draw()  # ensure positions are current
    axpos = ax.get_position()
    slider_h = 0.045
    slider_gap = 0.11     # how far below the axes the slider sits
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
        current_iter = min(nIters - 1, current_iter + 1)
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

    # Keyboard shortcuts (optional)
    def on_key(event):
        if event.key in (" ", "space"):
            do_play()
        elif event.key in ("right", "n", "j"):
            do_next()
        elif event.key in ("left", "p", "k"):
            do_prev()

    fig.canvas.mpl_connect("key_press_event", on_key)

    anim = FuncAnimation(fig, update, frames=max(1, nIters), init_func=init,
                         interval=args.interval, repeat=True, blit=False)

    if args.save:
        ext = args.save.lower().split(".")[-1]
        fps = max(1, 1000 // args.interval)
        if ext == "mp4":
            anim.save(args.save, writer="ffmpeg", fps=fps)
        elif ext == "gif":
            anim.save(args.save, writer="imagemagick", fps=fps)
        else:
            print("Unknown extension; use .mp4 or .gif. Showing window instead.")
            plt.show()
    else:
        plt.show()

if __name__ == "__main__":
    main()

