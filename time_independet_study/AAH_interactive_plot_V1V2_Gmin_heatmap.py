"""
Local desktop UI (tkinter + matplotlib): G_min(V1, V2) and phi_0(V1, V2) heatmaps
plus the phason-angle energy spectrum E(phi) at a chosen (V1, V2) point.

Needs AAH_model.py, AAH_tools.py and the updated AAH_tid_functions.py (with
GminPhi0Map / _gmin_row / GminPhi0 / EnergyPhiSpectrum / AutoWindingLevel) in the
same folder.

The winding state (lvl_i) can be typed in, or found automatically: enter a
(V1, V2) point under "Auto-select winding state at (V1, V2)" and press
"Auto-select winding state". The state is found for the current N and theta and
filled into lvl_i; the heatmaps still cover the whole (V1, V2) grid.

After pressing Calculate you can:
  * click on the left heatmap (G_min), or
  * type V1, V2 by hand and press "Plot spectrum" (or Enter in those fields)
to see E(phi) for that point on the right. The two levels whose gap is shown in the
heatmap (lvl_i, lvl_i + 1) are highlighted, and the phason angle phi_0 of the minimal
gap is marked with a dashed line.
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
N_PHI_SPECTRUM = 400 # number of phi points in the spectrum plot
T_HOP = 1.0
N_WORKERS = None     # None = all CPU cores
DATA_PATH = "data/Gmin_phi0"   # results are cached here (same files as the compute script)
AUTO_V1_DEFAULT = "0.5"        # default (V1, V2) point for the auto-selection fields
AUTO_V2_DEFAULT = "1.0"
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

        # ---- automatic selection of the winding state
        ttk.Label(panel, text="Auto-select winding state at (V1, V2):",
                  wraplength=230).grid(row=4, column=0, columnspan=2, sticky="w",
                                       pady=(12, 0))
        for r, (label, key, default) in enumerate(
                [("V1", "auto_V1", AUTO_V1_DEFAULT), ("V2", "auto_V2", AUTO_V2_DEFAULT)],
                start=5):
            ttk.Label(panel, text=label).grid(row=r, column=0, sticky="w", pady=2)
            v = tk.StringVar(value=default)
            ttk.Entry(panel, textvariable=v, width=10).grid(row=r, column=1, padx=(8, 0))
            self.vars[key] = v
        self.auto_btn = ttk.Button(panel, text="Auto-select winding state",
                                   command=self.on_auto)
        self.auto_btn.grid(row=7, column=0, columnspan=2, sticky="ew", pady=(4, 0))

        self.log_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(panel, text="LOG_GMIN (log colour scale for G_min)",
                        variable=self.log_var, command=self._redraw
                        ).grid(row=8, column=0, columnspan=2, sticky="w", pady=(10, 0))

        self.btn = ttk.Button(panel, text="Calculate", command=self.on_calculate)
        self.btn.grid(row=9, column=0, columnspan=2, sticky="ew", pady=(16, 4))
        self.cancel_btn = ttk.Button(panel, text="Cancel", command=self.on_cancel,
                                     state="disabled")
        self.cancel_btn.grid(row=10, column=0, columnspan=2, sticky="ew")
        self.progress = ttk.Progressbar(panel, mode="determinate", maximum=100)
        self.progress.grid(row=11, column=0, columnspan=2, sticky="ew", pady=(10, 4))
        self.status = ttk.Label(panel, text="Set parameters and press Calculate.",
                                wraplength=230, foreground="gray")
        self.status.grid(row=12, column=0, columnspan=2, sticky="w")
        ttk.Label(panel, foreground="gray", wraplength=230,
                  text=f"Grid: {N_V1}×{N_V2} points, V ∈ [{V_MIN}, {V_MAX}]. "
                       f"Set in the code."
                  ).grid(row=13, column=0, columnspan=2, sticky="w", pady=(10, 0))

        # ---- spectrum point selection
        ttk.Separator(panel).grid(row=14, column=0, columnspan=2, sticky="ew", pady=14)
        ttk.Label(panel, text="Energy spectrum at (V1, V2)",
                  font=("TkDefaultFont", 10, "bold")
                  ).grid(row=15, column=0, columnspan=2, sticky="w")
        for r, (label, key) in enumerate([("V1", "V1"), ("V2", "V2")], start=16):
            ttk.Label(panel, text=label).grid(row=r, column=0, sticky="w", pady=4)
            v = tk.StringVar(value="")
            e = ttk.Entry(panel, textvariable=v, width=10)
            e.grid(row=r, column=1, padx=(8, 0))
            e.bind("<Return>", lambda ev: self.on_plot_spectrum())
            self.vars[key] = v
        self.spec_btn = ttk.Button(panel, text="Plot spectrum",
                                   command=self.on_plot_spectrum, state="disabled")
        self.spec_btn.grid(row=18, column=0, columnspan=2, sticky="ew", pady=(6, 4))
        ttk.Label(panel, foreground="gray", wraplength=230,
                  text="…or click on the G_min heatmap (left plot). "
                       "Disable the toolbar zoom/pan tool to click."
                  ).grid(row=19, column=0, columnspan=2, sticky="w")

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
    def on_auto(self):
        """Fill in lvl_i with the winding state found at the chosen (V1, V2) point."""
        if self.busy:
            return
        try:
            N = int(self.vars["N"].get())
            theta = float(self.vars["theta"].get())
            V1 = float(self.vars["auto_V1"].get())
            V2 = float(self.vars["auto_V2"].get())
        except ValueError:
            messagebox.showerror("Invalid input",
                                 "N, θ and the auto-select V1, V2 must be valid numbers.")
            return
        if N < 4:
            messagebox.showerror("Invalid input", "N must be at least 4.")
            return
        self.status.config(text="Searching for the winding state…", foreground="gray")
        self.update_idletasks()
        try:
            lvl = aah_fun.AutoWindingLevel(N=N, V1=V1, V2=V2, theta=theta, t=T_HOP)
        except Exception as e:
            self.status.config(text="Error.", foreground="red")
            messagebox.showerror("Auto-select failed", str(e))
            return
        self.vars["lvl_i"].set(str(lvl))
        self.status.config(
            text=f"Winding state at (V1, V2) = ({V1:g}, {V2:g}): "
                 f"lvl_i = {lvl}, lvl_i + 1 = {lvl + 1}.", foreground="gray")

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
        self.auto_btn.config(state="disabled")
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
        self.auto_btn.config(state="normal")
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
