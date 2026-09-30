"""
Local desktop UI (tkinter + matplotlib) for G(phi) and dIPR(phi) of the AAH chain.
No Streamlit / browser needed. Run with python or your IDE's Run button, from the
folder containing AAH_model.py, AAH_tools.py and AAH_tid_functions.py.
"""
import os
import sys
import threading
import queue
import tkinter as tk
from tkinter import ttk, messagebox

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

import AAH_tid_functions as aah_fun


def set_pi_ticks(ax):
    """x data are phi/pi, so ticks sit at multiples of 0.5."""
    ax.set_xticks(np.arange(0, 2.01, 0.5))
    ax.set_xticklabels(["0", r"$\pi/2$", r"$\pi$", r"$3\pi/2$", r"$2\pi$"])
    ax.set_xlim(0, 2)


def compute(N, V1, V2, theta, lvl_i, lvl_j, n_phi, t=1.0):
    phi_lin = np.linspace(0, 2 * np.pi, n_phi)
    energies, angles = aah_fun.EnergyPhiSpectrum(
        N=N, V1=V1, V2=V2, t=t, theta=theta, phi_lin=phi_lin, save=False)
    G, angles = aah_fun.G_Phi(
        energies=energies, angles=angles, level_i=lvl_i, level_j=lvl_j,
        N=N, V1=V1, V2=V2, theta=theta, t=t, save=False)
    # same convention as the original script: dIPR is taken for level lvl_j
    dipr, _ = aah_fun.dIPR_Phi(
        N=N, V1=V1, V2=V2, phi_lin=phi_lin, theta=theta, t=t,
        level_i=lvl_j, save=False)
    return G, dipr, angles


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("AAH chain: G(φ) and dIPR(φ)")
        self.geometry("1250x620")
        self.q = queue.Queue()
        self.busy = False

        # ---------------- left panel: inputs
        panel = ttk.Frame(self, padding=12)
        panel.pack(side=tk.LEFT, fill=tk.Y)

        self.vars = {}
        fields = [
            ("N (chain length)", "N", "34"),
            ("V1 (on-site)", "V1", "1.5"),
            ("V2 (hopping)", "V2", "1.5"),
            ("θ (phason offset, rad)", "theta", "0.7"),
            ("Winding state number (lvl_i)", "lvl_i", "20"),
            ("φ points", "n_phi", "5000"),
        ]
        for r, (label, key, default) in enumerate(fields):
            ttk.Label(panel, text=label).grid(row=r * 1, column=0, sticky="w", pady=4)
            v = tk.StringVar(value=default)
            ttk.Entry(panel, textvariable=v, width=10).grid(row=r, column=1, padx=(8, 0))
            self.vars[key] = v

        self.lvl_label = ttk.Label(panel, text="", foreground="gray")
        self.lvl_label.grid(row=len(fields), column=0, columnspan=2, sticky="w")
        self.vars["lvl_i"].trace_add("write", lambda *_: self._update_lvl_label())
        self._update_lvl_label()

        self.btn = ttk.Button(panel, text="Calculate", command=self.on_calculate)
        self.btn.grid(row=len(fields) + 1, column=0, columnspan=2, sticky="ew", pady=(16, 4))
        self.status = ttk.Label(panel, text="Set parameters and press Calculate.",
                                wraplength=220, foreground="gray")
        self.status.grid(row=len(fields) + 2, column=0, columnspan=2, sticky="w")
        self.bind("<Return>", lambda e: self.on_calculate())

        # ---------------- right: plots
        right = ttk.Frame(self)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.fig = Figure(figsize=(11, 5))
        self.ax_g = self.fig.add_subplot(121)
        self.ax_d = self.fig.add_subplot(122)
        self.canvas = FigureCanvasTkAgg(self.fig, master=right)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        NavigationToolbar2Tk(self.canvas, right).update()
        self._blank()

        self.after(100, self._poll)

    # ---------------- helpers
    def _update_lvl_label(self):
        try:
            i = int(self.vars["lvl_i"].get())
            self.lvl_label.config(text=f"lvl_i = {i}, lvl_j = {i + 1}")
        except ValueError:
            self.lvl_label.config(text="")

    def _blank(self):
        for ax in (self.ax_g, self.ax_d):
            ax.clear()
            ax.set_xlabel(r"$\phi$")
            set_pi_ticks(ax)
        self.ax_g.set_ylabel("G/t"); self.ax_g.set_title(r"G($\phi$)")
        self.ax_d.set_ylabel("dIPR"); self.ax_d.set_title(r"dIPR($\phi$)")
        self.fig.tight_layout()
        self.canvas.draw()

    def _read_params(self):
        N = int(self.vars["N"].get())
        V1 = float(self.vars["V1"].get())
        V2 = float(self.vars["V2"].get())
        theta = float(self.vars["theta"].get())
        lvl_i = int(self.vars["lvl_i"].get())
        n_phi = int(self.vars["n_phi"].get())
        lvl_j = lvl_i + 1
        if N < 4:
            raise ValueError("N must be at least 4.")
        if not (0 <= lvl_i and lvl_j <= N - 1):
            raise ValueError(f"Need 0 ≤ lvl_i and lvl_i + 1 ≤ N − 1 (= {N - 1}).")
        if n_phi < 10:
            raise ValueError("φ points must be at least 10.")
        return N, V1, V2, theta, lvl_i, lvl_j, n_phi

    # ---------------- actions
    def on_calculate(self):
        if self.busy:
            return
        try:
            params = self._read_params()
        except ValueError as e:
            messagebox.showerror("Invalid input", str(e))
            return
        self.busy = True
        self.btn.config(state="disabled")
        self.status.config(text="Calculating…", foreground="gray")

        def work():
            try:
                self.q.put(("ok", params, compute(*params)))
            except Exception as e:  # report errors in the GUI
                self.q.put(("err", params, e))

        threading.Thread(target=work, daemon=True).start()

    def _poll(self):
        try:
            kind, params, payload = self.q.get_nowait()
        except queue.Empty:
            self.after(100, self._poll)
            return
        self.busy = False
        self.btn.config(state="normal")
        if kind == "err":
            self.status.config(text="Error.", foreground="red")
            messagebox.showerror("Calculation failed", str(payload))
        else:
            self._draw(params, *payload)
            self.status.config(text="Done.", foreground="gray")
        self.after(100, self._poll)

    def _draw(self, params, G, dipr, angles):
        N, V1, V2, theta, lvl_i, lvl_j, _ = params
        for ax, y, yl, title in ((self.ax_g, G, "G/t", r"G($\phi$)"),
                                 (self.ax_d, dipr, "dIPR", r"dIPR($\phi$)")):
            ax.clear()
            ax.plot(angles, y, label=fr"$\theta$={theta}")
            set_pi_ticks(ax)
            ax.set_xlabel(r"$\phi$")
            ax.set_ylabel(yl)
            ax.set_title(title)
            ax.legend(loc="right")
        self.fig.suptitle(f"N={N}, t=1.0, V1={V1}, V2={V2}, levels ({lvl_i}, {lvl_j})",
                          y=0.03, fontsize=10)
        self.fig.tight_layout(rect=(0, 0.05, 1, 1))
        self.canvas.draw()


if __name__ == "__main__":
    App().mainloop()