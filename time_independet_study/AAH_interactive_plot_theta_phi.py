# =============================================================================
# README
# =============================================================================
# AAH chain: (theta, phi) heatmaps of the gap G and of the dIPR
#
# WHAT IT DOES
#   Opens a local window that maps, for a finite Aubry-Andre-Harper (AAH) chain
#   with fixed V1, V2, over a grid of the phason offset theta and the phason
#   angle phi (both in [0, 2*pi]):
#     - G(theta, phi)    = |E[lvl_j] - E[lvl_i]| / t   (left heatmap)
#                          the energy gap between two chosen levels
#     - dIPR(theta, phi) = directional inverse participation ratio of the
#                          eigenstate lvl_j            (right heatmap)
#                          (AAH_model.AAH_chain.dIPR, sign = direction of the
#                          localisation, shifted by the "dIPR parameter")
#   Both maps are obtained from a single diagonalisation per (theta, phi)
#   point. The axes of the plots are theta/pi and phi/pi.
#
# REQUIREMENTS
#   - Python 3 with tkinter, numpy, scipy, matplotlib and tqdm installed
#   - AAH_model.py, AAH_tools.py and the UPDATED AAH_tid_functions.py in the
#     SAME folder as this script. All the calculations (building the chain,
#     diagonalisation, G, dIPR, saving / loading of the data) are done by
#     AAH_tid_functions.ThetaPhiMap (with ThetaPhiMapFilename, LoadThetaPhiMap
#     and the worker _theta_row); this file contains only the user interface.
#
# HOW TO RUN
#   Run this file with Python (e.g. `python AAH_theta_phi_plotter.py`) or with
#   your IDE's Run button, from the folder that contains the files above.
#
# HOW TO USE
#   1. Fill in the parameters in the left panel:
#        N              chain length (integer, >= 2)
#        V1             amplitude of the on-site modulation
#        V2             amplitude of the hopping modulation
#        t              hopping amplitude (default 1.0)
#        lvl_i          index of the lower level of the pair (0 <= lvl_i <=
#                       N - 2). The second level is always lvl_j = lvl_i + 1:
#                       G is the gap between lvl_i and lvl_i + 1, and the dIPR
#                       is calculated for the state lvl_j = lvl_i + 1 (shown
#                       under the field).
#        dIPR parameter shift of the dIPR direction weight (default 0.3,
#                       see AAH_model.dIPR)
#        N_PHI, N_THETA number of grid points along phi and theta (default 200)
#        Data folder    folder where results are saved / loaded from
#   2. Optionally tick "Force recalculation" to ignore a saved file and
#      recompute.
#   3. Press "Calculate" (or the Enter key). The progress bar and status text
#      show how many theta rows of the grid are done. "Cancel" aborts a
#      running calculation.
#   4. Inspect the plots. The dIPR is always shown on a symmetric colour scale
#      (limits -max|dIPR| .. +max|dIPR|), so zero is the centre of the red-blue
#      map. The toolbar below the plots allows zooming, panning and saving the
#      figure as an image.
#
# SAVED DATA
#   Results are saved to the chosen data folder as one .npz file per set of
#   parameters, named
#     G_dIPR_ThetaPhi_N{N}_V1{V1}_V2{V2}_t{t}_lvl{i}-{j}_d{dparameter}_{N_THETA}x{N_PHI}.npz
#   If a file for exactly the current parameters (including the grid size)
#   already exists, it is loaded instead of recomputed (the status line says
#   "Loaded from saved file"). Tick "Force recalculation" or delete the file to
#   recompute. The files contain the keys G_matrix, dIPR_matrix (shape
#   (N_THETA, N_PHI): rows = theta, columns = phi), phi_lin, theta_lin, N, V1,
#   V2, t, level_i, level_j and dparameter - the same keys as the files made by
#   the data notebook, so they can be plotted with the plotting notebook too.
#
# SETTINGS IN THE CODE (not in the UI)
#   N_WORKERS      number of worker processes (None = all CPU cores)
#   DEFAULT_FOLDER default value of the data folder field
#
# NOTES
#   - The calculation (AAH_tid_functions.ThetaPhiMap) is parallelised over rows
#     of theta with multiple processes. The same function can be called from a
#     notebook / script, e.g.
#       aah_fun.ThetaPhiMap(N=34, V1=0.5, V2=1.0, level_i=20, level_j=21,
#                           data_path="data/theta_phi")
#     and it will use / create the same saved files as this window. Cost grows as N_THETA * N_PHI * N^3, so fine grids or large N
#     can take a few minutes.
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

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

import AAH_tid_functions as aah_fun

# ------------------------- settings (set here, not in the UI) ---------------
N_WORKERS = None                       # None = all CPU cores
DEFAULT_FOLDER = "data/theta_phi"      # default folder for saved results
# ----------------------------------------------------------------------------

Cancelled = aah_fun.Cancelled


def run_calculation(p, progress_cb, cancel_evt):
    """
    p: dict with N, V1, V2, t, lvl_i, lvl_j, dparameter, n_phi, n_theta, folder, force.
    Calculates (or loads from the data folder) the maps with AAH_tid_functions.ThetaPhiMap.
    """
    # "spawn" is safer than "fork" in a process that has Tk loaded
    res = aah_fun.ThetaPhiMap(
        N=p["N"], V1=p["V1"], V2=p["V2"], level_i=p["lvl_i"], level_j=p["lvl_j"],
        t=p["t"], dparameter=p["dparameter"], n_phi=p["n_phi"], n_theta=p["n_theta"],
        n_workers=N_WORKERS, data_path=p["folder"], save=True, force=p["force"],
        progress_cb=progress_cb, cancel_evt=cancel_evt,
        mp_context=multiprocessing.get_context("spawn"))
    return dict(G=res["G_matrix"], dIPR=res["dIPR_matrix"],
                phi=res["phi_lin"], theta=res["theta_lin"], cached=res["cached"])


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("AAH chain: G(θ, φ) and dIPR(θ, φ)")
        self.geometry("1350x640")
        self.q = queue.Queue()
        self.busy = False
        self.cancel_evt = threading.Event()
        self.result = None      # dict with results + the parameters used

        # ---------------- left panel: inputs
        panel = ttk.Frame(self, padding=12)
        panel.pack(side=tk.LEFT, fill=tk.Y)

        self.vars = {}
        fields = [("N (chain length)", "N", "34"),
                  ("V1 (on-site modulation)", "V1", "0.5"),
                  ("V2 (hopping modulation)", "V2", "1.0"),
                  ("t (hopping)", "t", "1.0"),
                  ("lvl_i (winding state number)", "lvl_i", "20"),
                  ("dIPR parameter", "dparameter", "0.3"),
                  ("N_PHI (φ points)", "n_phi", "200"),
                  ("N_THETA (θ points)", "n_theta", "200")]
        for r, (label, key, default) in enumerate(fields):
            ttk.Label(panel, text=label).grid(row=r, column=0, sticky="w", pady=3)
            v = tk.StringVar(value=default)
            ttk.Entry(panel, textvariable=v, width=10).grid(row=r, column=1, padx=(8, 0))
            self.vars[key] = v

        r = len(fields)
        self.lvl_label = ttk.Label(panel, text="", foreground="gray")
        self.lvl_label.grid(row=r, column=0, columnspan=2, sticky="w")
        self.vars["lvl_i"].trace_add("write", lambda *_: self._update_lvl_label())
        self._update_lvl_label()
        r += 1
        ttk.Label(panel, text="Data folder").grid(row=r, column=0, columnspan=2,
                                                   sticky="w", pady=(8, 0))
        self.vars["folder"] = tk.StringVar(value=DEFAULT_FOLDER)
        ttk.Entry(panel, textvariable=self.vars["folder"], width=30
                  ).grid(row=r + 1, column=0, columnspan=2, sticky="ew")

        self.force_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(panel, text="Force recalculation (ignore saved file)",
                        variable=self.force_var
                        ).grid(row=r + 2, column=0, columnspan=2, sticky="w", pady=(10, 0))

        self.btn = ttk.Button(panel, text="Calculate", command=self.on_calculate)
        self.btn.grid(row=r + 3, column=0, columnspan=2, sticky="ew", pady=(14, 4))
        self.cancel_btn = ttk.Button(panel, text="Cancel", command=self.on_cancel,
                                     state="disabled")
        self.cancel_btn.grid(row=r + 4, column=0, columnspan=2, sticky="ew")
        self.progress = ttk.Progressbar(panel, mode="determinate", maximum=100)
        self.progress.grid(row=r + 5, column=0, columnspan=2, sticky="ew", pady=(10, 4))
        self.status = ttk.Label(panel, text="Set parameters and press Calculate.",
                                wraplength=250, foreground="gray")
        self.status.grid(row=r + 6, column=0, columnspan=2, sticky="w")
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
            self.lvl_label.config(text=f"G: lvl_i = {i} and lvl_i + 1 = {i + 1}; "
                                       f"dIPR of level {i + 1}")
        except ValueError:
            self.lvl_label.config(text="")

    def _blank(self):
        self.fig.clear()
        for k, title in enumerate([r"$G(\theta,\phi)$", r"dIPR$(\theta,\phi)$"]):
            ax = self.fig.add_subplot(1, 2, k + 1)
            ax.set_xlabel(r"$\phi\ /\ \pi$"); ax.set_ylabel(r"$\theta\ /\ \pi$")
            ax.set_title(title)
            ax.set_xlim(0, 2); ax.set_ylim(0, 2)
        self.fig.tight_layout()
        self.canvas.draw()

    def _read_params(self):
        v = self.vars
        try:
            p = dict(N=int(v["N"].get()), V1=float(v["V1"].get()),
                     V2=float(v["V2"].get()), t=float(v["t"].get()),
                     lvl_i=int(v["lvl_i"].get()),
                     dparameter=float(v["dparameter"].get()),
                     n_phi=int(v["n_phi"].get()), n_theta=int(v["n_theta"].get()),
                     folder=v["folder"].get().strip(), force=self.force_var.get())
        except ValueError:
            raise ValueError("All parameters must be valid numbers "
                             "(N, lvl_i and grid sizes must be integers).")
        if p["N"] < 2:
            raise ValueError("N must be at least 2.")
        if p["t"] == 0:
            raise ValueError("t must be non-zero.")
        if not (0 <= p["lvl_i"] <= p["N"] - 2):
            raise ValueError(f"Need 0 ≤ lvl_i ≤ N − 2 (= {p['N'] - 2}).")
        p["lvl_j"] = p["lvl_i"] + 1      # the second level is always lvl_i + 1
        if p["n_phi"] < 2 or p["n_theta"] < 2:
            raise ValueError("N_PHI and N_THETA must be at least 2.")
        if not p["folder"]:
            raise ValueError("Data folder must not be empty.")
        return p

    # ---------------- actions
    def on_calculate(self):
        if self.busy:
            return
        try:
            p = self._read_params()
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
                res = run_calculation(p, progress_cb, self.cancel_evt)
                res["params"] = p
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
        p = r["params"]
        phi = r["phi"] / np.pi
        theta = r["theta"] / np.pi

        self.fig.clear()
        ax1 = self.fig.add_subplot(1, 2, 1)
        ax2 = self.fig.add_subplot(1, 2, 2)

        m1 = ax1.pcolormesh(phi, theta, r["G"], shading="auto", cmap="viridis")
        self.fig.colorbar(m1, ax=ax1).set_label(r"$G/t$")
        ax1.set_title(rf"$G_{{{p['lvl_i']},{p['lvl_j']}}}(\theta,\phi)$")

        kw = {}
        vmax = float(np.max(np.abs(r["dIPR"])))     # symmetric colour scale, zero in the centre
        if vmax > 0:
            kw = dict(vmin=-vmax, vmax=vmax)
        m2 = ax2.pcolormesh(phi, theta, r["dIPR"], shading="auto", cmap="RdBu", **kw)
        self.fig.colorbar(m2, ax=ax2).set_label("dIPR")
        ax2.set_title(f"dIPR (level {p['lvl_j']})")

        for ax in (ax1, ax2):
            ax.set_xlabel(r"$\phi\ /\ \pi$")
            ax.set_ylabel(r"$\theta\ /\ \pi$")
        self.fig.suptitle(f"N={p['N']}, V1={p['V1']:g}, V2={p['V2']:g}, t={p['t']:g}, "
                          f"dIPR parameter={p['dparameter']:g}",
                          y=0.03, fontsize=10)
        self.fig.tight_layout(rect=(0, 0.05, 1, 1))
        self.canvas.draw()


if __name__ == "__main__":   # required for multiprocessing on Windows / macOS
    App().mainloop()