import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np

# ── Problem parameters ─────────────────────────────────────────────────
T = 10.0
delta = 1.0


# ── Smooth component functions ─────────────────────────────────────────
def f(t):
    return np.sin(1.5 * t) + 2.0


def g(t):
    return np.exp(-0.05 * t)


# ── Shared breakpoint grid ─────────────────────────────────────────────
# bp covers [0, T+δ] and serves two purposes:
#   1. the only allowed values for a and b  (restricted to bp ∩ [0, T])
#   2. the left edges of the piecewise-constant ("step") h
N_BP = 51
bp = np.linspace(0, T + delta, N_BP)  # shape (N_BP,)
dt = bp[1] - bp[0]

# Stepwise h: evaluate original h = f·g at each breakpoint.
# h is then CONSTANT on [bp[k], bp[k+1]) with value h_step[k].
h_step = f(bp) * g(bp)  # shape (N_BP,)


# ── Piecewise-constant Riemann integral ────────────────────────────────
def step_integral(fn, lo, hi):
    """
    Integrate a step function from lo to hi.
    The step function equals fn(bp[k]) on the interval [bp[k], bp[k+1]).
    fn is called once per breakpoint interval that overlaps [lo, hi].
    """
    total = 0.0
    for k in range(N_BP - 1):
        left = max(bp[k], lo)
        right = min(bp[k + 1], hi)
        if right > left:
            total += fn(bp[k]) * (right - left)
    # Tail beyond the last breakpoint (only relevant if hi > bp[-1])
    if hi > bp[-1]:
        total += fn(bp[-1]) * (hi - bp[-1])
    return total


# H(x): cumulative step-integral of h from 0 to x
def H(x):
    return step_integral(lambda t: f(t) * g(t), 0.0, x)


H_total = H(T + delta)


# ── Objective: three-term integral with domain stretching ─────────────
def objective(a, b):
    # Term 1 – ∫₀ᵃ  h(t) dt
    term1 = H(a)

    # Term 2 – ∫ₐᵃ⁺ᵇ⁺δ  f(mapped)·g(t)·scale dt
    #           mapped = (t − a)·scale + a,  scale = b/(b+δ)
    if b > 1e-10:
        scale = b / (b + delta)
        term2 = step_integral(
            lambda t: f((t - a) * scale + a) * g(t) * scale,
            a,
            a + b + delta,
        )
    else:
        term2 = 0.0

    # Term 3 – ∫ₐ₊ᵦ₊δᵀ⁺δ  f(t−δ)·g(t) dt
    term3 = step_integral(
        lambda t: f(t - delta) * g(t),
        a + b + delta,
        T + delta,
    )

    return term1 + term2 + term3


# ── Discrete candidate sets for a and b ───────────────────────────────
a_choices = bp[bp <= T]  # breakpoints inside [0, T]
b_choices = bp[bp <= T]
Na = len(a_choices)
Nb = len(b_choices)

# ── Grid search over all feasible (a, b) pairs ─────────────────────────
A_grid, B_grid = np.meshgrid(a_choices, b_choices, indexing="ij")
feasible = (A_grid + B_grid) <= T

J = np.full((Na, Nb), np.nan)
for i, a in enumerate(a_choices):
    for j, b in enumerate(b_choices):
        if feasible[i, j]:
            J[i, j] = objective(a, b)

# ── Minimiser ──────────────────────────────────────────────────────────
idx = np.unravel_index(np.nanargmin(J), J.shape)
a_star = a_choices[idx[0]]
b_star = b_choices[idx[1]]
J_star = J[idx]

print(f"Breakpoints : N = {N_BP},  dt = {dt:.4f},  grid [0, {T + delta:.1f}]")
print(
    f"Candidates  : {Na} values for a,  {Nb} for b,  {int(feasible.sum())} feasible pairs"
)
print(f"a*          = {a_star:.4f}")
print(f"b*          = {b_star:.4f}")
print(f"J*          = {J_star:.6f}")
if b_star > 1e-10:
    print(f"(b*+δ)/b*   = {(b_star + delta) / b_star:.4f}")
else:
    print("(b*+δ)/b*   = ∞  (b* = 0)")

# ══════════════════════════════════════════════════════════════════════
#  Plots
# ══════════════════════════════════════════════════════════════════════

t_dense = np.linspace(0, T + delta, 800)
f_dense = f(t_dense)
g_dense = g(t_dense)


# Stepwise h on the dense grid — each point snaps to its bp interval value
def h_dense_step(t_arr):
    k = np.clip(np.searchsorted(bp, t_arr, side="right") - 1, 0, N_BP - 1)
    return h_step[k]


h_dense = h_dense_step(t_dense)

# Piecewise-linear H on the dense grid (exact for the step h)
H_dense = np.array([H(t) for t in t_dense])

WIN_COLOR = "crimson"
WIN_ALPHA = 0.13


def annotate_window(ax, y_data, a, b, color=WIN_COLOR):
    """
    Shade [a, a+b], draw dashed boundary lines, and add:
      • labels 'a* = X.XX' and 'a*+b* = X.XX' above each boundary
      • a double-headed arrow spanning b* labelled inside the window
    """
    ymin_d = float(np.min(y_data))
    ymax_d = float(np.max(y_data))
    yr = ymax_d - ymin_d or 1.0

    ax.axvspan(a, a + b, alpha=WIN_ALPHA, color=color, zorder=0)
    ax.axvline(a, color=color, lw=1.1, ls="--", zorder=2)
    ax.axvline(a + b, color=color, lw=1.1, ls="--", zorder=2)

    # Labels above the boundary lines (clip_on=False lets them sit above axes)
    y_top = ymax_d + 0.06 * yr
    ax.text(
        a,
        y_top,
        f"$a^*\\!=\\!{a:.2f}$",
        ha="center",
        va="bottom",
        fontsize=8,
        color="dimgray",
        clip_on=False,
    )
    ax.text(
        a + b,
        y_top,
        f"$a^*\\!+\\!b^*\\!=\\!{a + b:.2f}$",
        ha="center",
        va="bottom",
        fontsize=8,
        color=color,
        clip_on=False,
    )

    # Double-headed arrow for the b* span
    y_arrow = ymin_d + 0.18 * yr
    ax.annotate(
        "",
        xy=(a + b, y_arrow),
        xytext=(a, y_arrow),
        arrowprops=dict(arrowstyle="<->", color=color, lw=1.3),
        zorder=3,
    )
    ax.text(
        a + b / 2,
        y_arrow + 0.05 * yr,
        f"$b^*\\!=\\!{b:.2f}$",
        ha="center",
        va="bottom",
        fontsize=8.5,
        color=color,
        zorder=3,
        bbox=dict(fc="white", ec="none", alpha=0.75, pad=1),
    )


fig = plt.figure(figsize=(15, 10))
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.60, wspace=0.42)

# ── Panel 1 · f(t) ─────────────────────────────────────────────────────
ax_f = fig.add_subplot(gs[0, 0])
ax_f.plot(t_dense, f_dense, color="steelblue", lw=1.8)
ax_f.axhline(0, color="gray", lw=0.6, ls=":")
annotate_window(ax_f, f_dense, a_star, b_star)
ax_f.set_title("$f(t)$", fontsize=11)
ax_f.set_xlabel("$t$")
ax_f.grid(True, alpha=0.25)

# ── Panel 2 · g(t) ─────────────────────────────────────────────────────
ax_g = fig.add_subplot(gs[0, 1])
ax_g.plot(t_dense, g_dense, color="darkorange", lw=1.8)
ax_g.axhline(0, color="gray", lw=0.6, ls=":")
annotate_window(ax_g, g_dense, a_star, b_star)
ax_g.set_title("$g(t)$", fontsize=11)
ax_g.set_xlabel("$t$")
ax_g.grid(True, alpha=0.25)

# ── Panel 3 · h(t) step + H(t) ────────────────────────────────────────
ax_h = fig.add_subplot(gs[0, 2])

# Faint vertical lines at each breakpoint to show the step grid
for edge in bp:
    ax_h.axvline(edge, color="seagreen", lw=0.4, ls=":", alpha=0.45, zorder=0)

ax_h.step(
    t_dense,
    h_dense,
    color="seagreen",
    lw=1.8,
    where="post",
    label="$h$ (step)",
    zorder=2,
)
ax_h.axhline(0, color="gray", lw=0.6, ls=":")
annotate_window(ax_h, h_dense, a_star, b_star)
ax_h.set_ylabel("$h(t)$", color="seagreen")
ax_h.tick_params(axis="y", labelcolor="seagreen")
ax_h.set_xlabel("$t$")

ax_H = ax_h.twinx()
ax_H.plot(t_dense, H_dense, color="purple", lw=1.4, ls="-.", label="$H(t)$")
ax_H.set_ylabel("$H(t)$", color="purple")
ax_H.tick_params(axis="y", labelcolor="purple")

lines = ax_h.get_lines()[:1] + ax_H.get_lines()[:1]
labels = [ln.get_label() for ln in lines]
ax_h.legend(lines, labels, fontsize=8, loc="upper right")
ax_h.set_title("$h(t)$ [step]  and  $H(t)$", fontsize=11)
ax_h.grid(True, alpha=0.20)

# ── Panel 4 · J(a, b) landscape ───────────────────────────────────────
ax_J = fig.add_subplot(gs[1, :2])

J_plot = np.where(feasible, J, np.nan)
vmin = float(np.nanmin(J_plot))
vmax = float(np.nanmax(J_plot))
levels = np.linspace(vmin, vmax, 40)

# pcolormesh handles NaN / masked cells cleanly on a coarse grid
pcm = ax_J.pcolormesh(
    a_choices,
    b_choices,
    J_plot.T,
    cmap="viridis",
    vmin=vmin,
    vmax=vmax,
    shading="nearest",
)

# Overlay contour lines where data exists
J_c = np.where(feasible, J, np.nan)
ax_J.contour(
    a_choices, b_choices, J_c.T, levels=12, colors="white", linewidths=0.4, alpha=0.5
)

# Infeasible region
ax_J.fill([0, T, T, 0], [T, 0, T, T], color="0.22", alpha=0.50, label="infeasible")
a_bnd = np.linspace(0, T, 300)
ax_J.plot(a_bnd, T - a_bnd, "w--", lw=1.4, label="$a+b=T$")

# Crosshairs from optimal point to both axes
ax_J.plot([a_star, a_star], [0, b_star], color=WIN_COLOR, lw=0.9, ls=":", alpha=0.9)
ax_J.plot([0, a_star], [b_star, b_star], color=WIN_COLOR, lw=0.9, ls=":", alpha=0.9)

# Optimal point
ax_J.scatter(
    [a_star],
    [b_star],
    color=WIN_COLOR,
    s=200,
    zorder=6,
    marker="*",
    label=f"$(a^*, b^*) = ({a_star:.2f},\\,{b_star:.2f})$",
)

# Highlight a* on x-axis and b* on y-axis with coloured tick labels
base_x = [v for v in range(0, int(T) + 1, 2) if abs(v - a_star) > 0.3]
base_y = [v for v in range(0, int(T) + 1, 2) if abs(v - b_star) > 0.3]

ax_J.set_xticks(sorted(base_x + [a_star]))
ax_J.set_xticklabels(
    [
        f"$a^*$\n{a_star:.2f}" if abs(v - a_star) < 0.01 else f"{int(v)}"
        for v in sorted(base_x + [a_star])
    ],
    fontsize=8.5,
)
ax_J.set_yticks(sorted(base_y + [b_star]))
ax_J.set_yticklabels(
    [
        f"$b^*$={b_star:.2f}" if abs(v - b_star) < 0.01 else f"{int(v)}"
        for v in sorted(base_y + [b_star])
    ],
    fontsize=8.5,
)
for lbl in ax_J.get_xticklabels():
    if "a^*" in lbl.get_text():
        lbl.set_color(WIN_COLOR)
        lbl.set_fontweight("bold")
for lbl in ax_J.get_yticklabels():
    if "b^*" in lbl.get_text():
        lbl.set_color(WIN_COLOR)
        lbl.set_fontweight("bold")

plt.colorbar(pcm, ax=ax_J, pad=0.02, label="$J(a, b)$")
ax_J.set_xlim(0, T)
ax_J.set_ylim(0, T)
ax_J.set_xlabel("$a$", fontsize=11)
ax_J.set_ylabel("$b$", fontsize=11)
ax_J.set_title("Objective landscape $J(a, b)$", fontsize=11)
ax_J.legend(fontsize=9, loc="upper right")

# ── Panel 5 · Slices through the optimum ──────────────────────────────
ax_sl = fig.add_subplot(gs[1, 2])

b_mask = a_star + b_choices <= T
ax_sl.plot(
    b_choices[b_mask],
    J[idx[0], b_mask],
    color="darkorange",
    lw=1.8,
    label=f"$J(a^*,\\,b)$",
)
ax_sl.axvline(b_star, color="darkorange", lw=1.1, ls="--")
ax_sl.scatter([b_star], [J_star], color="darkorange", s=70, zorder=5)

a_mask = a_choices + b_star <= T
ax_sl.plot(
    a_choices[a_mask],
    J[a_mask, idx[1]],
    color="steelblue",
    lw=1.8,
    label=f"$J(a,\\,b^*)$",
)
ax_sl.axvline(a_star, color="steelblue", lw=1.1, ls="--")
ax_sl.scatter([a_star], [J_star], color="steelblue", s=70, zorder=5)

# Label the vertical markers
sl_yvals = np.concatenate([J[idx[0], b_mask], J[a_mask, idx[1]]])
sl_ymax = float(np.nanmax(sl_yvals))
sl_yr = sl_ymax - float(np.nanmin(sl_yvals)) or 1.0
ax_sl.text(
    b_star + 0.1,
    sl_ymax - 0.04 * sl_yr,
    f"$b^*\\!=\\!{b_star:.2f}$",
    va="top",
    fontsize=8,
    color="darkorange",
)
ax_sl.text(
    a_star + 0.1,
    sl_ymax - 0.16 * sl_yr,
    f"$a^*\\!=\\!{a_star:.2f}$",
    va="top",
    fontsize=8,
    color="steelblue",
)

ax_sl.set_xlabel("$a$  or  $b$", fontsize=11)
ax_sl.set_title("Slices through $(a^*,\\,b^*)$", fontsize=11)
ax_sl.legend(fontsize=9)
ax_sl.grid(True, alpha=0.25)

# ── Supertitle ─────────────────────────────────────────────────────────
ratio_str = (
    f"$(b^*\\!+\\!\\delta)/b^* = {(b_star + delta) / b_star:.4f}$"
    if b_star > 1e-10
    else "$(b^*+\\delta)/b^* = \\infty$"
)
fig.suptitle(
    f"Gradient-free grid search  ·  $N_{{\\mathrm{{bp}}}}={N_BP}$ breakpoints"
    f"  ·  $T={T}$,  $\\delta={delta}$\n"
    f"$a^* = {a_star:.4f}$,   $b^* = {b_star:.4f}$,"
    f"   $J^* = {J_star:.6f}$,   {ratio_str}",
    fontsize=11,
    y=1.01,
)

plt.savefig("optimizer_demo.pdf", bbox_inches="tight")
plt.savefig("optimizer_demo.png", dpi=150, bbox_inches="tight")
plt.show()
print("Figures saved → optimizer_demo.pdf / .png")
