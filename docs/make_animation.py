"""Animate a real abcopt run on the 2-D Rastrigin function, phase by phase.

Usage:  python docs/make_animation.py   (writes docs/abc_animation.gif)
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from abcopt import CanonicalABCSolver

B = 5.12
def rastrigin(x):
    return float(20 + np.sum(x ** 2 - 10 * np.cos(2 * np.pi * x)))

frames = []  # (phase, foods, trial, best, changed_mask, scout_idx, evals)

class Recorder(CanonicalABCSolver):
    def _snap(self, phase, before=None, scout=None):
        changed = np.zeros(self.SN, bool) if before is None else np.any(self.foods != before, axis=1)
        frames.append((phase, self.foods.copy(), self.trial.copy(),
                       None if self.best_params is None else self.best_params.copy(),
                       changed, scout, self._evals, self.best_cost))
    def _employed_bees(self):
        b = self.foods.copy(); super()._employed_bees(); self._snap("employed", b)
    def _onlooker_bees(self):
        b = self.foods.copy(); super()._onlooker_bees(); self._memorize_best(); self._snap("onlooker", b)
    def _scout_bee(self):
        worst = int(np.argmax(self.trial)); will = self.trial[worst] >= self.limit and not self._stop()
        b = self.foods.copy(); super()._scout_bee()
        self._snap("scout", b, scout=worst if will else None)

s = Recorder(rastrigin, 2, [-B, -B], [B, B], pop_size=10, max_evals=760, limit=12, random_state=3)
s.solve()

xs = np.linspace(-B, B, 400); X, Y = np.meshgrid(xs, xs)
Z = 20 + X**2 - 10*np.cos(2*np.pi*X) + Y**2 - 10*np.cos(2*np.pi*Y)

COL = {"employed": "#E8A33D", "onlooker": "#3B7DD8", "scout": "#D64545"}
TITLE = {"employed": "1 · Employed bees: each source tries one neighbour move",
         "onlooker": "2 · Onlooker bees: better sources get more extra tries",
         "scout": "3 · Scout bee: an exhausted source is abandoned"}

fig, ax = plt.subplots(figsize=(7.2, 6.0), dpi=100)
fig.patch.set_facecolor("white")

def draw(k):
    ax.clear()
    phase, foods, trial, best, changed, scout, ev, bc = frames[k]
    ax.contourf(X, Y, Z, levels=24, cmap="Greys", alpha=0.28)
    ax.plot(0, 0, marker="*", ms=16, color="#2a9d4b", mec="white", zorder=5)
    ax.annotate("global minimum", (0, 0), (0.4, -0.9), fontsize=8, color="#2a9d4b")
    prev = frames[k - 1][1] if k > 0 else foods
    for i in range(len(foods)):
        if changed[i] and scout != i:
            ax.annotate("", foods[i], prev[i], arrowprops=dict(arrowstyle="-|>", color=COL[phase], lw=2.2, mutation_scale=14))
    ax.scatter(foods[:, 0], foods[:, 1], s=110,
               c=["#D64545" if i == scout else ("#F2C14E" if changed[i] else "#ffffff") for i in range(len(foods))],
               edgecolors="#333333", linewidths=1.2, zorder=4)
    for i in range(len(foods)):
        ax.text(foods[i, 0] + 0.22, foods[i, 1] + 0.18, f"{int(trial[i])}", fontsize=7.5, color="#555555", zorder=7)
    if scout is not None:
        ax.scatter(*prev[scout], s=160, marker="x", c="#D64545", linewidths=2.5, zorder=6)
        ax.annotate("", foods[scout], prev[scout], arrowprops=dict(arrowstyle="-|>", color="#D64545", lw=1.8, ls="--", mutation_scale=14))
        ax.text(0.5, 1.012, "trial counter reached the limit → jump to a random new spot", transform=ax.transAxes,
                ha="center", va="bottom", fontsize=8.5, color="#D64545")
    if best is not None:
        ax.scatter(*best, s=260, facecolors="none", edgecolors="#2a9d4b", linewidths=2.2, zorder=6)
    cyc = k // 3 + 1
    ax.set_title(TITLE[phase], fontsize=11.5, color=COL[phase], loc="left", fontweight="bold", pad=16)
    ax.text(0.01, -0.09, f"cycle {cyc}   ·   evaluations {ev}   ·   best f = {bc:.3f}", transform=ax.transAxes, fontsize=9.5, color="#333333")
    ax.text(0.99, -0.09, "abcopt · 2-D Rastrigin", transform=ax.transAxes, fontsize=8.5, color="#777777", ha="right")
    from matplotlib.lines import Line2D
    h = [Line2D([], [], ls="", marker="o", mfc="#ffffff", mec="#333", ms=8, label="food source (number = failed trials)"),
         Line2D([], [], ls="", marker="o", mfc="#F2C14E", mec="#333", ms=8, label="improved in this phase"),
         Line2D([], [], ls="", marker="o", mfc="none", mec="#2a9d4b", mew=2, ms=11, label="best solution so far")]
    ax.legend(handles=h, loc="lower left", fontsize=7.5, framealpha=0.92)
    ax.set_xlim(-B, B); ax.set_ylim(-B, B); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values(): sp.set_color("#999999")

# keep first 18 cycles, then every 2nd cycle for the rest
keep = []
for k in range(len(frames)):
    cyc = k // 3
    if cyc < 18 or cyc % 3 == 0 or k >= len(frames) - 3:
        keep.append(k)
keep += [keep[-1]] * 8  # pause at the end
anim = FuncAnimation(fig, lambda i: draw(keep[i]), frames=len(keep))
fig.subplots_adjust(left=0.02, right=0.98, top=0.90, bottom=0.11)
anim.save(os.path.join(os.path.dirname(os.path.abspath(__file__)), "abc_animation.gif"), writer=PillowWriter(fps=3))
print(len(frames) // 3, "cycles,", len(keep), "frames, final best", s.best_cost)
