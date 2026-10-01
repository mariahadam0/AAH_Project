# =============================================================================
# README
# =============================================================================
# AAH chain: number of gap closings versus (V1, V2), a small desktop GUI
#
# WHAT IT DOES
#   Opens a local window (tkinter + matplotlib, no browser needed) that maps,
#   over a grid of (V1, V2) values for a finite Aubry-Andre-Harper (AAH) chain,
#   how many times the gap between two neighbouring energy levels closes as the
#   phason angle phi runs over [0, 2*pi):
#     - Gap:         G(phi) = (E[lvl_i+1] - E[lvl_i]) / t
#     - Gap closing: a local minimum of G(phi) that goes below GAP_TOL
#   Each candidate minimum is refined numerically, so sharp cusps are caught
#   even on a coarse phi grid. The result is a single heatmap with a discrete
#   colour bar showing the number of closings (0, 1, 2, ...) at each (V1, V2).
#   The calculation uses AAH_tid_functions.CountGapClosings (via _closing_row).
#
# REQUIREMENTS
#   - Python 3 with tkinter, numpy, matplotlib and scipy installed
#   - The following files in the SAME folder as this script:
#       AAH_model.py, AAH_tools.py and the UPDATED AAH_tid_functions.py
#       (it must contain CountGapClosings, _closing_row and ClosingMapFilename)
#
# HOW TO RUN
#   Run this file with Python (e.g. `python <this_file>.py`) or with your IDE's
#   Run button, from the folder that contains the files above.
#
# HOW TO USE
#   1. Fill in the parameters in the left panel:
#        N       chain length (integer, >= 4)
#        theta   phason offset in radians
#        lvl_i   index of the lower level of the pair (0 <= lvl_i <= N - 2);
#                the gap is taken between lvl_i and lvl_i + 1 (shown under the
#                fields)
#   2. Press "Calculate" (or the Enter key). The progress bar and status text
#      show how many rows of the grid are done. "Cancel" aborts a running
#      calculation.
#   3. Inspect the plot. The toolbar below it allows zooming, panning and
#      saving the figure as an image.
#
# SETTINGS IN THE CODE (not in the UI)
#   V_MIN, V_MAX   range of both V1 and V2 (default 0 to 2)
#   N_V1, N_V2     grid resolution along V1 and V2 (default 81 x 81)
#   N_PHI          number of phi points per (V1, V2) point (default 900);
#                  closings closer than about 2 grid cells can merge into one
#   GAP_TOL        gap/t below this value counts as "closed" (default 1e-3)
#   T_HOP          hopping amplitude t (default 1.0)
#   N_WORKERS      number of worker processes (None = all CPU cores)
#   DATA_PATH      folder where results are cached (default "data/gap_closings")
#
# NOTES
#   - The calculation is parallelised over rows of V2 with multiple processes
#     and can take a few minutes for fine grids or large N.
#   - Results are saved as .npz files in DATA_PATH. If a file matching the
#     current N, theta, lvl_i, grid, N_PHI, t and GAP_TOL already exists, it is
#     loaded instead of recomputed. Change any of those settings to force a new
#     calculation, or delete the file.
#   - Invalid input is reported in a pop-up dialog; calculation errors are also
#     shown in a pop-up.
# =============================================================================


import os
import sys
import queue
import threading
import multiprocessing
import tkinter as tk
from tkinter import ttk, messagebox
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.colors import BoundaryNorm
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

import AAH_tid_functions as aah_fun

# ------------------------- settings (set here, not in the UI) ---------------
V_MIN, V_MAX = 0.0, 2.0
N_V1 = 81            # resolution of the V1 axis
N_V2 = 81            # resolution of the V2 axis
N_PHI = 900          # phi points per (V1, V2) point (closings closer than ~2 cells can merge)
GAP_TOL = 1e-3       # gap/t below this counts as "closed"
T_HOP = 1.0
N_WORKERS = None     # None = all CPU cores
DATA_PATH = "data/gap_closings"   # results are cached here
# ----------------------------------------------------------------------------


class Cancelled(Exception):
    pass


def run_calculation(N, theta, lvl_i, progress_cb, cancel_evt):
    V1_lin = np.linspace(V_MIN, V_MAX, N_V1)
    V2_lin = np.linspace(V_MIN, V_MAX, N_V2)
    path = os.path.join(DATA_PATH, aah_fun.ClosingMapFilename(N, theta, lvl_i))

    if os.path.exists(path):
        d = np.load(path)
        if (d["V1_lin"].shape == V1_lin.shape and d["V2_lin"].shape == V2_lin.shape
                and np.allclose(d["V1_lin"], V1_lin) and np.allclose(d["V2_lin"], V2_lin)
                and int(d["n_phi"]) == N_PHI and float(d["t"]) == T_HOP
                and float(d["gap_tol"]) == GAP_TOL):
            return dict(n=d["n_matrix"], V1=V1_lin, V2=V2_lin, cached=True)

    n_map = np.zeros((N_V2, N_V1), dtype=int)
    jobs = [(r, V2, V1_lin, N, lvl_i, theta, T_HOP, N_PHI, GAP_TOL)
            for r, V2 in enumerate(V2_lin)]

    # "spawn" is safer than "fork" in a process that has Tk loaded
    ctx = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=N_WORKERS or os.cpu_count() or 1,
                             mp_context=ctx) as ex:
        futures = [ex.submit(aah_fun._closing_row, job) for job in jobs]
        done = 0
        for f in as_completed(futures):
            if cancel_evt.is_set():
                for g in futures:
                    g.cancel()
                raise Cancelled()
            r, row = f.result()
            n_map[r] = row
            done += 1
            progress_cb(done, len(jobs))

    os.makedirs(DATA_PATH, exist_ok=True)
    np.savez(path, n_matrix=n_map, V1_lin=V1_lin, V2_lin=V2_lin, N=N, theta=theta,
             t=T_HOP, level_i=lvl_i, n_phi=N_PHI, gap_tol=GAP_TOL)
    return dict(n=n_map, V1=V1_lin, V2=V2_lin, cached=False)


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("AAH chain: number of gap closings vs (V1, V2)")
        self.geometry("1050x650")
        self.q = queue.Queue()
        self.busy = False
        self.cancel_evt = threading.Event()
        self.result = None

        # ---------------- left panel
        panel = ttk.Frame(self, padding=12)
        panel.pack(side=tk.LEFT, fill=tk.Y)

        self.vars = {}
        fields = [("N (chain length)", "N", "34"),
                  ("θ (phason offset, rad)", "theta", "0.0"),
                  ("Winding state number (lvl_i)", "lvl_i", "20")]
        for r, (label, key, default) in enumerate(fields):
            ttk.Label(panel, text=label).grid(row=r, column=0, sticky="w", pady=4)
            v = tk.StringVar(value=default)
            ttk.Entry(panel, textvariable=v, width=10).grid(row=r, column=1, padx=(8, 0))
            self.vars[key] = v

        self.lvl_label = ttk.Label(panel, text="", foreground="gray")
        self.lvl_label.grid(row=3, column=0, columnspan=2, sticky="w")
        self.vars["lvl_i"].trace_add("write", lambda *_: self._update_lvl_label())
        self._update_lvl_label()

        self.btn = ttk.Button(panel, text="Calculate", command=self.on_calculate)
        self.btn.grid(row=4, column=0, columnspan=2, sticky="ew", pady=(16, 4))
        self.cancel_btn = ttk.Button(panel, text="Cancel", command=self.on_cancel,
                                     state="disabled")
        self.cancel_btn.grid(row=5, column=0, columnspan=2, sticky="ew")
        self.progress = ttk.Progressbar(panel, mode="determinate", maximum=100)
        self.progress.grid(row=6, column=0, columnspan=2, sticky="ew", pady=(10, 4))
        self.status = ttk.Label(panel, text="Set parameters and press Calculate.",
                                wraplength=230, foreground="gray")
        self.status.grid(row=7, column=0, columnspan=2, sticky="w")

        ttk.Label(panel, foreground="gray", wraplength=230,
                  text=f"Counts the φ values where the gap closes (gap/t < {GAP_TOL:g}). "
                       f"Grid: {N_V1}×{N_V2} points, V ∈ [{V_MIN}, {V_MAX}], "
                       f"{N_PHI} φ points. Set in the code."
                  ).grid(row=8, column=0, columnspan=2, sticky="w", pady=(14, 0))
        self.bind("<Return>", lambda e: self.on_calculate())

        # ---------------- right: plot
        right = ttk.Frame(self)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.fig = Figure(figsize=(7, 6))
        self.canvas = FigureCanvasTkAgg(self.fig, master=right)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        NavigationToolbar2Tk(self.canvas, right).update()
        self._blank()

        self.after(100, self._poll)

    # ---------------- helpers
    def _update_lvl_label(self):
        try:
            i = int(self.vars["lvl_i"].get())
            self.lvl_label.config(text=f"gap between lvl_i = {i} and lvl_i + 1 = {i + 1}")
        except ValueError:
            self.lvl_label.config(text="")

    def _blank(self):
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        ax.set_xlabel("V1"); ax.set_ylabel("V2")
        ax.set_xlim(V_MIN, V_MAX); ax.set_ylim(V_MIN, V_MAX)
        ax.set_aspect("equal")
        self.fig.tight_layout()
        self.canvas.draw()

    def _read_params(self):
        N = int(self.vars["N"].get())
        theta = float(self.vars["theta"].get())
        lvl_i = int(self.vars["lvl_i"].get())
        if N < 4:
            raise ValueError("N must be at least 4.")
        if not (0 <= lvl_i and lvl_i + 1 <= N - 1):
            raise ValueError(f"Need 0 ≤ lvl_i and lvl_i + 1 ≤ N − 1 (= {N - 1}).")
        return N, theta, lvl_i

    # ---------------- actions
    def on_calculate(self):
        if self.busy:
            return
        try:
            N, theta, lvl_i = self._read_params()
        except ValueError as e:
            messagebox.showerror("Invalid input", str(e))
            return
        self.busy = True
        self.cancel_evt.clear()
        self.btn.config(state="disabled")
        self.cancel_btn.config(state="normal")
        self.progress.config(value=0)
        self.status.config(text="Calculating…", foreground="gray")

        def progress_cb(done, total):
            self.q.put(("progress", done, total))

        def work():
            try:
                res = run_calculation(N, theta, lvl_i, progress_cb, self.cancel_evt)
                res.update(N=N, theta=theta, lvl_i=lvl_i)
                self.q.put(("ok", res, None))
            except Cancelled:
                self.q.put(("cancelled", None, None))
            except Exception as e:
                self.q.put(("err", e, None))

        threading.Thread(target=work, daemon=True).start()

    def on_cancel(self):
        self.cancel_evt.set()
        self.status.config(text="Cancelling…")

    def _finish(self):
        self.busy = False
        self.btn.config(state="normal")
        self.cancel_btn.config(state="disabled")

    def _poll(self):
        try:
            while True:
                kind, a, b = self.q.get_nowait()
                if kind == "progress":
                    self.progress.config(value=100 * a / b)
                    self.status.config(text=f"Calculating… {a}/{b} rows")
                elif kind == "ok":
                    self._finish()
                    self.result = a
                    self.progress.config(value=100)
                    self.status.config(text="Loaded from saved file." if a["cached"]
                                       else "Done (saved to file).", foreground="gray")
                    self._draw()
                elif kind == "cancelled":
                    self._finish()
                    self.progress.config(value=0)
                    self.status.config(text="Cancelled.")
                elif kind == "err":
                    self._finish()
                    self.status.config(text="Error.", foreground="red")
                    messagebox.showerror("Calculation failed", str(a))
        except queue.Empty:
            pass
        self.after(100, self._poll)

    def _draw(self):
        r = self.result
        Z = r["n"]
        vmax = int(Z.max())
        cmap = matplotlib.colormaps["viridis"].resampled(vmax + 1)
        norm = BoundaryNorm(np.arange(-0.5, vmax + 1.5), cmap.N)

        self.fig.clear()
        ax = self.fig.add_subplot(111)
        m = ax.pcolormesh(r["V1"], r["V2"], Z, shading="nearest", cmap=cmap, norm=norm)
        cb = self.fig.colorbar(m, ax=ax, ticks=np.arange(0, vmax + 1))
        cb.set_label("number of gap closings")
        ax.set_xlabel("V1"); ax.set_ylabel("V2")
        ax.set_title(r"Number of gap closings in $\phi\in[0, 2\pi)$")
        ax.set_aspect("equal")
        self.fig.suptitle(fr"N={r['N']}, t={T_HOP}, $\theta$={r['theta']}, "
                          f"levels ({r['lvl_i']}, {r['lvl_i'] + 1})",
                          y=0.03, fontsize=10)
        self.fig.tight_layout(rect=(0, 0.05, 1, 1))
        self.canvas.draw()


if __name__ == "__main__":   # required for multiprocessing on Windows / macOS
    App().mainloop()