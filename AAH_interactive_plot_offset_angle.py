"""
Interactive explorer for the AAH energy spectrum as a function of k (theta).

Workflow
--------
1. Set N, V1, V2, phi (in radians), t and click "Compute spectrum".
   The (thetas, energies) pair is cached, so re-clicking the same
   parameters (or just tweaking delta / selection) never re-diagonalizes.
2. Click on any energy line in the plot to select it (click again to
   deselect). Up to two levels can be selected at once.
3. "Search winding" runs find_winding_states to auto-select the pair of
   levels with the largest winding gap.
4. "Find theta_min" needs exactly two selected levels; it runs find_min_gap
   and draws a vertical line at the smallest-gap theta.
5. The delta slider draws theta_min +/- delta/2 as two dashed lines. Moving
   it only redraws lines (no recomputation), so it stays instant.
6. "Optimize delta" scans a fixed, code-level grid of delta values
   (see DELTA_SEARCH_MIN/MAX/RES below) with AAH_tools.find_max_ipr_delta
   to find the delta that maximizes the IPR of the first selected level,
   then moves the delta slider there.
7. Click "Compute IPR & PD" to get the probability density (and IPR) of
   the *first* selected level, evaluated at both theta_1 and theta_2, plotted
   together on one pop-up figure. Requires theta_min to have been found
   first (theta_1/theta_2 are derived from it via the delta slider).

Assumes this file lives next to AAH_tools.py, with the AAH_model package
importable as `from AAH import AAH_model`.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import TextBox, Button, Slider

from AAH import AAH_model as aah_model
import AAH_tools as aah_tools


class AAH_UI:

    NK = 1000  # resolution of the theta (k) sweep from 0 to 2*pi

    # Fixed search settings for the "Optimize delta" button (not user-editable).
    DELTA_SEARCH_MIN = 0.0
    DELTA_SEARCH_MAX = np.pi
    DELTA_SEARCH_RES = 300

    def __init__(self):
        self.cache = {}          # {(N,V1,V2,phi,t): (thetas, energies)}
        self.N = None
        self.V1 = self.V2 = self.phi = self.t = None
        self.thetas = None       # theta/pi values, shape (NK,)
        self.energies = None     # list of eigenvalue arrays, one per theta angle
        self.lines = []          # Line2D per energy level, index = level
        self.selected_levels = []  # up to 2 level indices

        self.theta_min = None
        self.theta_1 = None
        self.theta_2 = None
        self.vline_min = None
        self.vline_1 = None
        self.vline_2 = None
        self.fig_pd = None       # pop-up probability-density figure

        self.fig = plt.figure(figsize=(13, 7))
        self.ax_spec = self.fig.add_axes([0.08, 0.10, 0.60, 0.83])
        self.ax_spec.set_xlabel(r'$\theta/\pi$')
        self.ax_spec.set_ylabel('E/t')
        self.ax_spec.grid(alpha=0.3)

        self._build_widgets()
        self.status_text = self.fig.text(0.08, 0.02, "Set parameters and click 'Compute spectrum'.",
                                          fontsize=9, color='0.2')

        self.on_compute(None)  # draw a default spectrum on startup
        self.fig.canvas.mpl_connect('pick_event', self.on_pick)

    # ------------------------------------------------------------------ UI

    def _build_widgets(self):
        x0, w = 0.74, 0.22

        ax_N = self.fig.add_axes([x0, 0.905, w, 0.045])
        ax_V1 = self.fig.add_axes([x0, 0.855, w, 0.045])
        ax_V2 = self.fig.add_axes([x0, 0.805, w, 0.045])
        ax_phi = self.fig.add_axes([x0, 0.755, w, 0.045])
        ax_t = self.fig.add_axes([x0, 0.705, w, 0.045])

        self.tb_N = TextBox(ax_N, 'N', initial='34')
        self.tb_V1 = TextBox(ax_V1, 'V1', initial='1.0')
        self.tb_V2 = TextBox(ax_V2, 'V2', initial='0.0')
        self.tb_phi = TextBox(ax_phi, r'$\phi$', initial='0.0')
        self.tb_t = TextBox(ax_t, 't', initial='1.0')

        ax_compute = self.fig.add_axes([x0, 0.635, w, 0.05])
        self.btn_compute = Button(ax_compute, 'Compute spectrum')
        self.btn_compute.on_clicked(self.on_compute)

        ax_search = self.fig.add_axes([x0, 0.57, w, 0.05])
        self.btn_search = Button(ax_search, 'Search winding state')
        self.btn_search.on_clicked(self.on_search)

        ax_thetamin = self.fig.add_axes([x0, 0.505, w, 0.05])
        self.btn_thetamin = Button(ax_thetamin, 'Find theta_min')
        self.btn_thetamin.on_clicked(self.on_find_theta_min)

        self._syncing_delta = False  # guards against slider<->textbox feedback loops

        ax_delta = self.fig.add_axes([x0, 0.45, w, 0.035])
        self.slider_delta = Slider(ax_delta, r'$\delta/\pi$', 0.0, 0.5, valinit=0.05)
        self.slider_delta.on_changed(self.on_delta_slider_changed)

        ax_delta_tb = self.fig.add_axes([x0 + w * 0.35, 0.40, w * 0.65, 0.04])
        self.tb_delta = TextBox(ax_delta_tb, 'value ', initial=f'{self.slider_delta.val:.4f}')
        self.tb_delta.on_submit(self.on_delta_text_submit)

        ax_opt_delta = self.fig.add_axes([x0, 0.32, w, 0.05])
        self.btn_opt_delta = Button(ax_opt_delta, 'Optimize delta (max IPR)')
        self.btn_opt_delta.on_clicked(self.on_optimize_delta)

        ax_ipr = self.fig.add_axes([x0, 0.19, w, 0.05])
        self.btn_ipr_pd = Button(ax_ipr, 'Compute IPR && PD (theta_1 & theta_2)')
        self.btn_ipr_pd.on_clicked(self.on_ipr_pd)

    # ---------------------------------------------------------- callbacks

    def on_compute(self, event):
        try:
            N = int(self.tb_N.text)
            V1 = float(self.tb_V1.text)
            V2 = float(self.tb_V2.text)
            phi = float(self.tb_phi.text)
            t = float(self.tb_t.text)
        except ValueError:
            self.update_status(r"Invalid parameter value(s) - check N, V1, V2, $\phi$, t.")
            return

        key = (N, round(V1, 6), round(V2, 6), round(phi, 6), round(t, 6))
        if key in self.cache:
            thetas, energies = self.cache[key]
            self.update_status("Loaded cached spectrum for these parameters.")
        else:
            thetas = np.linspace(0, 2, self.NK, endpoint=True)  # theta/pi
            energies = []
            for th in thetas:
                chain = aah_model.AAH_chain(N=N, t=t, V1=V1, V2=V2, theta=th * np.pi, phi=phi)
                eigvals, _ = chain.diagonalize()
                energies.append(eigvals)
            self.cache[key] = (thetas, energies)
            self.update_status("Computed a new spectrum (now cached).")

        self.N, self.V1, self.V2, self.phi, self.t = N, V1, V2, phi, t
        self.thetas = thetas
        self.energies = energies
        self.selected_levels = []
        self.theta_min = self.theta_1 = self.theta_2 = None
        self.draw_spectrum()

    def draw_spectrum(self):
        self.ax_spec.cla()
        self.lines = []
        for i in range(self.N):
            y = [self.energies[j][i] for j in range(len(self.thetas))]
            line, = self.ax_spec.plot(self.thetas, y, color='0.6', lw=0.8, picker=True)
            line.set_pickradius(5)
            line.level_idx = i
            self.lines.append(line)

        self.ax_spec.set_xlabel(r'$\theta/\pi$')
        self.ax_spec.set_ylabel('E/t')
        self.ax_spec.set_title(
            f'N={self.N}, V1={self.V1:g}, V2={self.V2:g}, $\\phi$={self.phi:g}, t={self.t:g}',
            fontsize=10)
        self.ax_spec.grid(alpha=0.3)
        self.vline_min = self.vline_1 = self.vline_2 = None
        self.fig.canvas.draw_idle()

    def on_pick(self, event):
        i = getattr(event.artist, 'level_idx', None)
        if i is None:
            return
        if i in self.selected_levels:
            self.selected_levels.remove(i)
        else:
            self.selected_levels.append(i)
            if len(self.selected_levels) > 2:
                self.selected_levels.pop(0)
        self.update_line_colors()
        self.update_status(f"Selected levels: {self.selected_levels}")

    def update_line_colors(self):
        palette = ['tab:red', 'tab:blue']
        for line in self.lines:
            line.set_color('0.6')
            line.set_linewidth(0.8)
            line.set_zorder(1)
        for rank, idx in enumerate(self.selected_levels):
            self.lines[idx].set_color(palette[rank])
            self.lines[idx].set_linewidth(2.0)
            self.lines[idx].set_zorder(3)
        self.fig.canvas.draw_idle()

    def on_search(self, event):
        if self.energies is None:
            self.update_status("Compute the spectrum first.")
            return
        states = aah_tools.find_winding_states(
            N=self.N, energies=self.energies, parameters=self.thetas,
            n_gaps=2, only_positive=True)
        self.selected_levels = states
        self.update_line_colors()
        self.update_status(f"Winding states found: {states}")

    def on_find_theta_min(self, event):
        if self.energies is None:
            self.update_status("Compute the spectrum first.")
            return
        if len(self.selected_levels) != 2:
            self.update_status("Select exactly 2 levels first (click lines, or use Search).")
            return
        li, lj = self.selected_levels
        j_star, theta_min = aah_tools.find_min_gap(
            N=self.N, energies=self.energies, parameters=self.thetas,
            level_i=li, level_j=lj)
        self.theta_min = theta_min

        if self.vline_min is not None:
            self.vline_min.remove()
        self.vline_min = self.ax_spec.axvline(
            theta_min / np.pi, color='k', linestyle='--', lw=1.2)
        self.update_status(f"theta_min = {theta_min:.4f} rad ({theta_min/np.pi:.4f} pi)")
        # self.update_status(f"theta_min = {theta_min:.4f} rad ({theta_min/np.pi} pi)")
        self.update_delta_lines()

    def on_optimize_delta(self, event):
        if self.energies is None:
            self.update_status("Compute the spectrum first.")
            return
        if not self.selected_levels:
            self.update_status("Select at least one level first.")
            return
        if self.theta_min is None:
            self.update_status("Find theta_min first.")
            return

        level = self.selected_levels[0]
        delta_list = np.linspace(
            self.DELTA_SEARCH_MIN, self.DELTA_SEARCH_MAX, self.DELTA_SEARCH_RES)

        # theta is the base value (theta_min) that delta/2 is added to;
        # phi stays fixed at the current UI value. vary="theta" tells
        # find_max_ipr_delta to perturb theta rather than phi.
        max_ipr, best_delta = aah_tools.find_max_ipr_delta(
            N=self.N, t=self.t, V1=self.V1, V2=self.V2,
            theta=self.theta_min, phi=self.phi,
            level=level, vary="theta", delta_list=delta_list)

        best_delta_over_pi = min(max(best_delta / np.pi, self.slider_delta.valmin),
                                  self.slider_delta.valmax)
        self.slider_delta.set_val(best_delta_over_pi)  # triggers on_delta_slider_changed

        self.update_status(
            f"state {level}: max IPR={max_ipr:.4f} at delta={best_delta:.4f} rad "
            f"({best_delta_over_pi:.4f} pi)")

    def on_delta_slider_changed(self, val):
        if self._syncing_delta:
            return
        self._syncing_delta = True
        try:
            self.tb_delta.set_val(f'{val:.4f}')
        finally:
            self._syncing_delta = False
        self.update_delta_lines(val)

    def on_delta_text_submit(self, text):
        if self._syncing_delta:
            return
        try:
            val = float(text)
        except ValueError:
            self.update_status("Invalid delta/pi value - enter a number.")
            return

        clipped = min(max(val, self.slider_delta.valmin), self.slider_delta.valmax)
        if clipped != val:
            self.update_status(
                f"delta/pi clamped to slider range [{self.slider_delta.valmin}, {self.slider_delta.valmax}].")

        self._syncing_delta = True
        try:
            self.slider_delta.set_val(clipped)  # moves the slider handle too
        finally:
            self._syncing_delta = False
        self.tb_delta.set_val(f'{clipped:.4f}')
        self.update_delta_lines(clipped)

    def update_delta_lines(self, val=None):
        if self.theta_min is None:
            self.fig.canvas.draw_idle()
            return
        delta = self.slider_delta.val * np.pi
        self.theta_1 = self.theta_min - delta / 2
        self.theta_2 = self.theta_min + delta / 2

        if self.vline_1 is not None:
            self.vline_1.remove()
        if self.vline_2 is not None:
            self.vline_2.remove()
        self.vline_1 = self.ax_spec.axvline(
            self.theta_1 / np.pi, color='tab:green', linestyle=':', lw=1.2)
        self.vline_2 = self.ax_spec.axvline(
            self.theta_2 / np.pi, color='tab:green', linestyle=':', lw=1.2)
        self.fig.canvas.draw_idle()

    def on_ipr_pd(self, event):
        if self.energies is None:
            self.update_status("Compute the spectrum first.")
            return
        if not self.selected_levels:
            self.update_status("Select at least one level first.")
            return
        if self.theta_1 is None or self.theta_2 is None:
            self.update_status("Find theta_min first - theta_1 & theta_2 are derived from it.")
            return

        level = self.selected_levels[0]

        chain_1 = aah_model.AAH_chain(N=self.N, t=self.t, V1=self.V1, V2=self.V2,
                                       theta=self.theta_1, phi=self.phi)
        chain_1.diagonalize()
        pd_1 = chain_1.probability_density()
        ipr_1 = chain_1.IPR()[level]

        chain_2 = aah_model.AAH_chain(N=self.N, t=self.t, V1=self.V1, V2=self.V2,
                                       theta=self.theta_2, phi=self.phi)
        chain_2.diagonalize()
        pd_2 = chain_2.probability_density()
        ipr_2 = chain_2.IPR()[level]

        if self.fig_pd is not None:
            plt.close(self.fig_pd)
        self.fig_pd, ax_pd = plt.subplots(figsize=(6, 4))
        sites = np.arange(1, self.N + 1)

        ax_pd.plot(sites, pd_1[level], marker='o', ms=3, color='tab:green',
                   label=f'theta_1 = {self.theta_1/np.pi:.4f} pi  (IPR={ipr_1:.4f})')
        ax_pd.plot(sites, pd_2[level], marker='s', ms=3, color='tab:purple',
                   label=f'theta_2 = {self.theta_2/np.pi:.4f} pi  (IPR={ipr_2:.4f})')

        ax_pd.set_xlabel('site n')
        ax_pd.set_ylabel(r'$|\psi_n|^2$')
        ax_pd.set_title(f'Probability density of state {level} at theta_1 & theta_2')
        ax_pd.legend()
        ax_pd.grid(alpha=0.3)
        self.fig_pd.show()

        self.update_status(
            f"state {level}: IPR(theta_1)={ipr_1:.4f}, IPR(theta_2)={ipr_2:.4f}")

    def update_status(self, msg):
        self.status_text.set_text(msg)
        self.fig.canvas.draw_idle()


ui = AAH_UI()
plt.show()