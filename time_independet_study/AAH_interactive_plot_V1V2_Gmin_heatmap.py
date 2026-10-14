# =============================================================================
# README
# =============================================================================
# AAH chain: G_min(V1, V2), phi_0(V1, V2) heatmaps and E(phi) spectrum
#
# WHAT IT DOES
#   Opens a local window that maps,
#   over a grid of (V1, V2) values for a finite Aubry-Andre-Harper (AAH) chain,
#   two quantities that depend on the phason angle phi:
#     - G_min(V1, V2): the minimal gap |E[lvl_i+1] - E[lvl_i]| / t between two
#                      neighbouring energy levels, minimised over phi
#                      (left heatmap)
#     - phi_0(V1, V2): the value of phi at which that minimum occurs
#                      (middle heatmap, colour bar labelled in multiples of pi)
#   For every (V1, V2) point, phi is first scanned coarsely (N_PHI points) and
#   the minimum is then refined, using AAH_tid_functions._gmin_row.
#
#   In addition, for any chosen point (V1, V2) the window shows the energy
#   spectrum E(phi) of the chain as a function of the phason angle phi
#   (right panel, next to the two heatmaps):
#     - all levels are drawn in light grey
#     - the two levels whose gap is mapped (lvl_i and lvl_i + 1) are highlighted
#       in blue and red
#     - a dashed vertical line marks phi_0, the phason angle of the minimal gap
#       (computed exactly for the chosen point with GminPhi0, not read off the
#       grid); the title shows V1, V2 and G_min / t
#     - a red "+" marks the chosen point on both heatmaps
#
# REQUIREMENTS
#   - Python 3 with tkinter, numpy and matplotlib installed
#   - The following files in the SAME folder as this script:
#       AAH_model.py, AAH_tools.py and the UPDATED AAH_tid_functions.py
#       (it must contain GminPhi0Map, _gmin_row, GminMapFilename, GminPhi0
#       and EnergyPhiSpectrum)
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
#   2. Optionally tick "LOG_GMIN" to show G_min on a logarithmic colour scale
#      (this can be toggled after the calculation without recomputing; the
#      selected point is kept).
#   3. Press "Calculate". The progress bar and status text show how many rows
#      of the grid are done. "Cancel" aborts a running calculation.
#   4. Choose a point (V1, V2) to see its energy spectrum E(phi), in either way:
#        - CLICK on the left heatmap (G_min). The V1 and V2 fields in the
#          "Energy spectrum at (V1, V2)" section are filled in with the clicked
#          values. Clicks are ignored while the toolbar zoom or pan tool is
#          active, so switch it off first.
#        - TYPE V1 and V2 into those fields and press "Plot spectrum" (or the
#          Enter key inside either field). Values do not have to lie on the
#          grid. They can be fine-tuned by hand after a click.
#      The spectrum panel is only available after a calculation has finished.
#      Pressing "Calculate" again clears the selected point, since N, theta or
#      lvl_i may have changed.
#   5. Inspect the plots. The toolbar below them allows zooming, panning and
#      saving the figure as an image.
#
# SETTINGS IN THE CODE (not in the UI)
#   V_MIN, V_MAX      range of both V1 and V2 (default 0 to 2; clicked points
#                     are clamped to this range)
#   N_V1, N_V2        grid resolution along V1 and V2 (default 81 x 81)
#   N_PHI             number of points in the coarse phi scan per grid point
#                     (also used when locating phi_0 of a chosen point)
#   N_PHI_SPECTRUM    number of phi points in the E(phi) spectrum plot
#                     (default 400)
#   T_HOP             hopping amplitude t (default 1.0)
#   N_WORKERS         number of worker processes (None = all CPU cores)
#   DATA_PATH         folder where results are cached (default "data/Gmin_phi0")
#
# NOTES
#   - The heatmap calculation is parallelised over rows of V2 with multiple
#     processes and can take a few minutes for fine grids or large N. The
#     spectrum of a single point is computed in the main process and is fast
#     (it is cached, so redrawing, e.g. toggling LOG_GMIN, does not recompute).
#   - Heatmap results are saved as .npz files in DATA_PATH. If a file matching
#     the current N, theta, lvl_i, grid, N_PHI and t already exists, it is
#     loaded instead of recomputed. Change any of those settings to force a new
#     calculation, or delete the file. Spectra are not saved to disk.
#   - Invalid input is reported in a pop-up dialog; calculation errors are also
#     shown in a pop-up. If the spectrum cannot be computed, the error is shown
#     inside the spectrum panel.
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
from matplotlib.colors import LogNorm
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

import AAH_tid_functions as aah_fun

# ------------------------- resolution (set here, not in the UI) -------------
V_MIN, V_MAX = 0.0, 2.0
N_V1 = 81            # resolution of the V1 axis
N_V2 = 81            # resolution of the V2 axis
N_PHI = 360          # coarse phi scan per (V1, V2) point (minimum is then refined)
N_PHI_SPECTRUM = 400 # number of phi points in the spectrum plot
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
        self.title("AAH chain: G_min(V1, V2), φ₀(V1, V2) and E(φ)")
        self.geometry("1850x650")
        self.q = queue.Queue()
        self.busy = False
        self.cancel_evt = threading.Event()
        self.result = None      # dict with results + the parameters used
        self.sel = None         # selected (V1, V2) point
        self.ax_gmin = None     # left heatmap axes (click target)
        self.spec_cache = None  # (key, phi/pi, energies, phi0 in units of pi)

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

        # ---- spectrum point selection
        ttk.Separator(panel).grid(row=10, column=0, columnspan=2, sticky="ew", pady=14)
        ttk.Label(panel, text="Energy spectrum at (V1, V2)",
                  font=("TkDefaultFont", 10, "bold")
                  ).grid(row=11, column=0, columnspan=2, sticky="w")
        for r, (label, key) in enumerate([("V1", "V1"), ("V2", "V2")], start=12):
            ttk.Label(panel, text=label).grid(row=r, column=0, sticky="w", pady=4)
            v = tk.StringVar(value="")
            e = ttk.Entry(panel, textvariable=v, width=10)
            e.grid(row=r, column=1, padx=(8, 0))
            e.bind("<Return>", lambda ev: self.on_plot_spectrum())
            self.vars[key] = v
        self.spec_btn = ttk.Button(panel, text="Plot spectrum",
                                   command=self.on_plot_spectrum, state="disabled")
        self.spec_btn.grid(row=14, column=0, columnspan=2, sticky="ew", pady=(6, 4))
        ttk.Label(panel, foreground="gray", wraplength=230,
                  text="…or click on the G_min heatmap (left plot). "
                       "Disable the toolbar zoom/pan tool to click."
                  ).grid(row=15, column=0, columnspan=2, sticky="w")

        # ---------------- right: plots
        right = ttk.Frame(self)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.fig = Figure(figsize=(17, 5.5))
        self.canvas = FigureCanvasTkAgg(self.fig, master=right)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.toolbar = NavigationToolbar2Tk(self.canvas, right)
        self.toolbar.update()
        self.canvas.mpl_connect("button_press_event", self._on_click)
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
        titles = [r"$G_{min}$(V1, V2)", r"$\phi_0$(V1, V2)", r"$E(\phi)$"]
        for k, title in enumerate(titles):
            ax = self.fig.add_subplot(1, 3, k + 1)
            ax.set_title(title)
            if k < 2:
                ax.set_xlabel("V1"); ax.set_ylabel("V2")
                ax.set_xlim(V_MIN, V_MAX); ax.set_ylim(V_MIN, V_MAX)
                ax.set_aspect("equal")
            else:
                ax.set_xlabel(r"$\phi/\pi$"); ax.set_ylabel("E")
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

    def on_plot_spectrum(self):
        """Read V1, V2 from the entry fields and plot the spectrum there."""
        if self.result is None:
            messagebox.showinfo("No data", "Press Calculate first.")
            return
        try:
            v1 = float(self.vars["V1"].get())
            v2 = float(self.vars["V2"].get())
        except ValueError:
            messagebox.showerror("Invalid input", "V1 and V2 must be numbers.")
            return
        self.sel = (v1, v2)
        self._redraw()

    def _on_click(self, event):
        if self.result is None or event.inaxes is not self.ax_gmin:
            return
        if self.toolbar.mode != "":      # zoom / pan active
            return
        if event.xdata is None or event.ydata is None:
            return
        v1 = float(np.clip(event.xdata, V_MIN, V_MAX))
        v2 = float(np.clip(event.ydata, V_MIN, V_MAX))
        self.vars["V1"].set(f"{v1:.4f}")
        self.vars["V2"].set(f"{v2:.4f}")
        self.sel = (v1, v2)
        self._redraw()

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
                    self.sel = None
                    self.spec_cache = None
                    self.spec_btn.config(state="normal")
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

    # ---------------- spectrum
    def _spectrum(self, v1, v2):
        r = self.result
        key = (r["N"], r["theta"], r["lvl_i"], round(v1, 10), round(v2, 10))
        if self.spec_cache is not None and self.spec_cache[0] == key:
            return self.spec_cache[1:]
        phi_lin = np.linspace(0, 2 * np.pi, N_PHI_SPECTRUM)
        energies, angles = aah_fun.EnergyPhiSpectrum(
            N=r["N"], V1=v1, V2=v2, phi_lin=phi_lin, t=T_HOP, theta=r["theta"], save=False)
        g_min, phi0 = aah_fun.GminPhi0(N=r["N"], V1=v1, V2=v2, level_i=r["lvl_i"],
                                       theta=r["theta"], t=T_HOP, n_phi=N_PHI)
        self.spec_cache = (key, angles, energies, g_min, phi0)
        return angles, energies, g_min, phi0

    def _draw_spectrum(self, ax):
        r = self.result
        ax.set_xlabel(r"$\phi/\pi$"); ax.set_ylabel("E")
        if self.sel is None:
            ax.set_title(r"$E(\phi)$")
            ax.text(0.5, 0.5, "Click on the G_min map\nor enter V1, V2\nand press 'Plot spectrum'",
                    ha="center", va="center", transform=ax.transAxes, color="gray")
            return
        v1, v2 = self.sel
        lvl = r["lvl_i"]
        angles, E, g_min, phi0 = self._spectrum(v1, v2)

        ax.plot(angles, E, color="0.7", lw=0.6)                      # all levels
        ax.plot(angles, E[:, lvl], color="tab:blue", lw=1.8, label=f"lvl {lvl}")
        ax.plot(angles, E[:, lvl + 1], color="tab:red", lw=1.8, label=f"lvl {lvl + 1}")
        if np.isfinite(phi0):
            ax.axvline(phi0, color="k", ls="--", lw=0.8,
                       label=fr"$\phi_0$={phi0:.3f}$\pi$")
        ax.set_xlim(angles[0], angles[-1])
        ax.set_title(fr"$E(\phi)$ at V1={v1:.3f}, V2={v2:.3f}" "\n"
                     fr"$G_{{min}}/t$={g_min / 1.0:.4g}", fontsize=10)
        ax.legend(loc="best", fontsize=8)

    # ---------------- drawing
    def _redraw(self):
        r = self.result
        if r is None:
            return
        self.fig.clear()
        ax1 = self.fig.add_subplot(1, 3, 1)
        ax2 = self.fig.add_subplot(1, 3, 2)
        ax3 = self.fig.add_subplot(1, 3, 3)
        self.ax_gmin = ax1

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
            if self.sel is not None:
                ax.plot(*self.sel, marker="+", color="red", ms=14, mew=2)

        try:
            self._draw_spectrum(ax3)
        except Exception as e:
            ax3.set_title(r"$E(\phi)$")
            ax3.text(0.5, 0.5, f"Spectrum failed:\n{e}", ha="center", va="center",
                     transform=ax3.transAxes, color="red")

        self.fig.suptitle(fr"N={r['N']}, t={T_HOP}, $\theta$={r['theta']}, "
                          f"levels ({r['lvl_i']}, {r['lvl_i'] + 1})",
                          y=0.03, fontsize=10)
        self.fig.tight_layout(rect=(0, 0.05, 1, 1))
        self.canvas.draw()


if __name__ == "__main__":   # required for multiprocessing on Windows / macOS
    App().mainloop()
