"""
Local desktop UI (tkinter + matplotlib): G_min(V1, V2) and phi_0(V1, V2) heatmaps.

Needs AAH_model.py, AAH_tools.py and the updated AAH_tid_functions.py (with
GminPhi0Map / _gmin_row) in the same folder. Run with python or your IDE's Run button.

G_min : minimal gap |E[lvl_i+1] - E[lvl_i]|/t over the phason angle phi
phi_0 : phason angle at which that minimum occurs
"""
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
from matplotlib.colors import LogNorm
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

import AAH_tid_functions as aah_fun

# ------------------------- resolution (set here, not in the UI) -------------
V_MIN, V_MAX = 0.0, 2.0
N_V1 = 81            # resolution of the V1 axis
N_V2 = 81            # resolution of the V2 axis
N_PHI = 360          # coarse phi scan per (V1, V2) point (minimum is then refined)
T_HOP = 1.0
N_WORKERS = None     # None = all CPU cores
DATA_PATH = "data/Gmin_phi0"   # results are cached here (same files as the compute script)
# ----------------------------------------------------------------------------


class Cancelled(Exception):
    pass


def run_calculation(N, theta, lvl_i, progress_cb, cancel_evt):
    """Returns dict with Gmin, phi0, V1_lin, V2_lin. Uses the .npz cache if it matches."""
    V1_lin = np.linspace(V_MIN, V_MAX, N_V1)
    V2_lin = np.linspace(V_MIN, V_MAX, N_V2)
    path = os.path.join(DATA_PATH, aah_fun.GminMapFilename(N, theta, lvl_i))

    if os.path.exists(path):
        d = np.load(path)
        if (d["V1_lin"].shape == V1_lin.shape and d["V2_lin"].shape == V2_lin.shape
                and np.allclose(d["V1_lin"], V1_lin) and np.allclose(d["V2_lin"], V2_lin)
                and int(d["n_phi"]) == N_PHI and float(d["t"]) == T_HOP):
            return dict(Gmin=d["Gmin_matrix"], phi0=d["phi0_matrix"],
                        V1=V1_lin, V2=V2_lin, cached=True)

    Gmin = np.zeros((N_V2, N_V1))
    phi0 = np.zeros_like(Gmin)
    jobs = [(r, V2, V1_lin, N, lvl_i, theta, T_HOP, N_PHI) for r, V2 in enumerate(V2_lin)]

    # "spawn" is safer than "fork" in a process that has Tk loaded
    ctx = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=N_WORKERS or os.cpu_count() or 1,
                             mp_context=ctx) as ex:
        futures = [ex.submit(aah_fun._gmin_row, job) for job in jobs]
        done = 0
        for f in as_completed(futures):
            if cancel_evt.is_set():
                for g in futures:
                    g.cancel()
                raise Cancelled()
            r, g, p = f.result()
            Gmin[r], phi0[r] = g, p
            done += 1
            progress_cb(done, len(jobs))

    os.makedirs(DATA_PATH, exist_ok=True)
    np.savez(path, Gmin_matrix=Gmin, phi0_matrix=phi0, V1_lin=V1_lin, V2_lin=V2_lin,
             N=N, theta=theta, t=T_HOP, level_i=lvl_i, n_phi=N_PHI)
    return dict(Gmin=Gmin, phi0=phi0, V1=V1_lin, V2=V2_lin, cached=False)


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("AAH chain: G_min(V1, V2) and φ₀(V1, V2)")
        self.geometry("1350x620")
        self.q = queue.Queue()
        self.busy = False
        self.cancel_evt = threading.Event()
        self.result = None      # dict with results + the parameters used

        # ---------------- left panel: inputs
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

        self.log_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(panel, text="LOG_GMIN (log colour scale for G_min)",
                        variable=self.log_var, command=self._redraw
                        ).grid(row=4, column=0, columnspan=2, sticky="w", pady=(10, 0))

        self.btn = ttk.Button(panel, text="Calculate", command=self.on_calculate)
        self.btn.grid(row=5, column=0, columnspan=2, sticky="ew", pady=(16, 4))
        self.cancel_btn = ttk.Button(panel, text="Cancel", command=self.on_cancel,
                                     state="disabled")
        self.cancel_btn.grid(row=6, column=0, columnspan=2, sticky="ew")
        self.progress = ttk.Progressbar(panel, mode="determinate", maximum=100)
        self.progress.grid(row=7, column=0, columnspan=2, sticky="ew", pady=(10, 4))
        self.status = ttk.Label(panel, text="Set parameters and press Calculate.",
                                wraplength=230, foreground="gray")
        self.status.grid(row=8, column=0, columnspan=2, sticky="w")
        ttk.Label(panel, foreground="gray", wraplength=230,
                  text=f"Grid: {N_V1}×{N_V2} points, V ∈ [{V_MIN}, {V_MAX}]. "
                       f"Set in the code."
                  ).grid(row=9, column=0, columnspan=2, sticky="w", pady=(10, 0))
        self.bind("<Return>", lambda e: self.on_calculate())

        # ---------------- right: plots
        right = ttk.Frame(self)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.fig = Figure(figsize=(12, 5.5))
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
        for k, title in enumerate([r"$G_{min}$(V1, V2)", r"$\phi_0$(V1, V2)"]):
            ax = self.fig.add_subplot(1, 2, k + 1)
            ax.set_xlabel("V1"); ax.set_ylabel("V2"); ax.set_title(title)
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
        self.status.config(text="Calculating… (may take a few minutes)", foreground="gray")

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
                                       else "Done (saved to file).")
                    self._redraw()
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

    def _redraw(self):
        r = self.result
        if r is None:
            return
        self.fig.clear()
        ax1 = self.fig.add_subplot(1, 2, 1)
        ax2 = self.fig.add_subplot(1, 2, 2)

        kw = {}
        if self.log_var.get():
            pos = r["Gmin"][r["Gmin"] > 0]
            if pos.size and r["Gmin"].max() > max(pos.min(), 1e-12):
                kw["norm"] = LogNorm(vmin=max(pos.min(), 1e-12), vmax=r["Gmin"].max())
        m1 = ax1.pcolormesh(r["V1"], r["V2"], r["Gmin"], shading="nearest",
                            cmap="viridis", **kw)
        self.fig.colorbar(m1, ax=ax1).set_label(r"$G_{min}/t$")
        ax1.set_title(r"$G_{min}$(V1, V2)")

        m2 = ax2.pcolormesh(r["V1"], r["V2"], r["phi0"], shading="nearest",
                            cmap="twilight", vmin=0, vmax=2)
        cb2 = self.fig.colorbar(m2, ax=ax2)
        cb2.set_label(r"$\phi_0$")
        cb2.set_ticks([0, 0.5, 1, 1.5, 2])
        cb2.set_ticklabels(["0", r"$\pi/2$", r"$\pi$", r"$3\pi/2$", r"$2\pi$"])
        ax2.set_title(r"$\phi_0$(V1, V2)")

        for ax in (ax1, ax2):
            ax.set_xlabel("V1"); ax.set_ylabel("V2"); ax.set_aspect("equal")
        self.fig.suptitle(fr"N={r['N']}, t={T_HOP}, $\theta$={r['theta']}, "
                          f"levels ({r['lvl_i']}, {r['lvl_i'] + 1})",
                          y=0.03, fontsize=10)
        self.fig.tight_layout(rect=(0, 0.05, 1, 1))
        self.canvas.draw()


if __name__ == "__main__":   # required for multiprocessing on Windows / macOS
    App().mainloop()