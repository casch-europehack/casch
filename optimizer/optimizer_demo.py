import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.integrate import quad
from scipy.optimize import minimize_scalar

# ------------------------------------------------------------
# Example data: f ≡ 1 (constant), g has 3 Gaussian peaks of width ≈ delta
# ------------------------------------------------------------
T = 22.0
delta = 0.5   # fixed extension amount

x_grid = np.linspace(0, T + 2.0, 6000)

# f: constant
f_vals = np.ones_like(x_grid)

# g: three Gaussian peaks with FWHM ≈ delta
#    FWHM = 2√(2 ln 2) · σ ≈ 2.355 σ  →  σ = delta / 2.355
peak_sigma = delta / 2.355
peak_centers = [2.5, 7.0, 11.5]
peak_amps    = [6.0, 5.0, 5.0]
g_vals = np.zeros_like(x_grid)
for mu, amp in zip(peak_centers, peak_amps):
    g_vals += amp * np.exp(-0.5 * ((x_grid - mu) / peak_sigma) ** 2)

f_fun = interp1d(x_grid, f_vals, kind='linear', bounds_error=False, fill_value=0.0)
g_fun = interp1d(x_grid, g_vals, kind='cubic', bounds_error=False, fill_value=0.0)

# ------------------------------------------------------------
# g-cropping objective
#   Given current g on [0, T_g], crop out [a, a+δ] and shift left:
#     g'(x) = g(x)           for x < a
#     g'(x) = g(x + delta)   for x ≥ a
#   New effective domain: [0, T_g - δ]
#
#   J(a) = ∫₀^{T_g - δ} f(x) · g'(x) dx
# ------------------------------------------------------------

def J_crop(a, f_func, g_func, T_g, delta):
    """Objective: integral of f · g' where g' has [a, a+δ] cropped out."""
    T_new = T_g - delta
    if a < 0 or a > T_new:
        return 1e12

    def integrand(x):
        if x < a:
            return float(f_func(x)) * float(g_func(x))
        else:
            return float(f_func(x)) * float(g_func(x + delta))

    return quad(integrand, 0, T_new, limit=200, points=[a])[0]


# ------------------------------------------------------------
# Mapping: compressed-domain position → original g-coordinate
#
# sorted_real_cuts: cut start positions in original g-space (sorted).
# After k cuts, compressed position x maps to original x + k'·δ
# where k' is the number of cuts whose original position ≤ the
# mapped value.  We resolve this iteratively.
# ------------------------------------------------------------

def compressed_to_real(x, sorted_real_cuts, delta):
    """Map compressed-space position to original g coordinate."""
    pos = x
    for k, c in enumerate(sorted_real_cuts):
        # In compressed space this cut appears at c - k*delta
        threshold = c - k * delta
        if pos >= threshold:
            pos += delta
        else:
            break
    return pos


# ============================================================
# Iterative optimisation: crop g  n_steps  times
# ============================================================
n_steps = 10
n_interp = 4000

# Working copy of g (will be rebuilt each iteration)
cur_x = np.linspace(0, T, n_interp)
cur_g_vals = np.array([float(g_fun(x)) for x in cur_x])
cur_g = interp1d(cur_x, cur_g_vals, kind='cubic',
                 bounds_error=False, fill_value=0.0)

T_g = T                  # effective domain length of current g
sorted_real_cuts = []     # cut starts in *original* g-space (kept sorted)
compressed_cuts  = []     # a* at each step (compressed space)
J_star_iter      = []

print("--- Iterative g-cropping optimisation ---")
for step_i in range(n_steps):
    T_new = T_g - delta   # domain after this crop

    # Capture loop variables for the closure
    _g = cur_g
    _T_g = T_g

    def J_this(a, _f=f_fun, _g=_g, _T_g=_T_g, _delta=delta):
        return J_crop(a, _f, _g, _T_g, _delta)

    res = minimize_scalar(J_this, bounds=(0.0, T_new),
                          method='bounded',
                          options={'xatol': 1e-10, 'maxiter': 1000})

    a_s = res.x
    compressed_cuts.append(a_s)
    J_star_iter.append(res.fun)

    # Map to original g-coordinate
    real_a = compressed_to_real(a_s, sorted_real_cuts, delta)
    sorted_real_cuts.append(real_a)
    sorted_real_cuts.sort()

    print(f"  Step {step_i+1}: a_compressed={a_s:.3f}, "
          f"a_real={real_a:.3f}, "
          f"J*={res.fun:.5f}, T_eff={T_new:.1f}")

    # Build the new (cropped) g on [0, T_new]
    new_x = np.linspace(0, T_new, n_interp)
    new_g_vals = np.zeros(n_interp)
    for k, xv in enumerate(new_x):
        if xv < a_s:
            new_g_vals[k] = float(cur_g(xv))
        else:
            new_g_vals[k] = float(cur_g(xv + delta))

    cur_g = interp1d(new_x, new_g_vals, kind='linear',
                     bounds_error=False, fill_value=0.0)
    T_g = T_new

print(f"\nReal cut positions (original g-space, sorted):")
for i, c in enumerate(sorted_real_cuts):
    print(f"  Cut {i+1}: [{c:.3f}, {c + delta:.3f}]")

# ============================================================
# Final visualisation — three vertically stacked plots
#   1) f(x)   with cropped-out intervals marked
#   2) g(x)   with cropped-out intervals marked
#   3) f·g    with cropped-out intervals marked
# ============================================================
x_plot = np.linspace(0, T, 2000)
f_plot = np.array([float(f_fun(x)) for x in x_plot])
g_plot = np.array([float(g_fun(x)) for x in x_plot])
fg_plot = f_plot * g_plot

gap_colors = ['#ff6666', '#6699ff', '#66cc66', '#ff9933', '#cc66ff']

fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)

# --- Plot 1: f with intervals ---
ax = axes[0]
ax.plot(x_plot, f_plot, color='tab:blue', linewidth=2, label=r'$f(x)$')
for gi, c in enumerate(sorted_real_cuts):
    col = gap_colors[gi % len(gap_colors)]
    ax.axvspan(c, c + delta, alpha=0.35, color=col,
               label=f'Cut {gi+1}: [{c:.2f}, {c+delta:.2f}]')
ax.set_ylabel(r'$f(x)$', fontsize=12)
ax.set_title(r'$f(x)$ with cropped-out intervals', fontsize=13)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# --- Plot 2: g with intervals ---
ax = axes[1]
ax.plot(x_plot, g_plot, color='tab:orange', linewidth=2, label=r'$g(x)$')
for gi, c in enumerate(sorted_real_cuts):
    col = gap_colors[gi % len(gap_colors)]
    ax.axvspan(c, c + delta, alpha=0.35, color=col,
               label=f'Cut {gi+1}: [{c:.2f}, {c+delta:.2f}]')
ax.set_ylabel(r'$g(x)$', fontsize=12)
ax.set_title(r'$g(x)$ with cropped-out intervals', fontsize=13)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# --- Plot 3: f·g with intervals ---
ax = axes[2]
ax.plot(x_plot, fg_plot, color='tab:green', linewidth=2,
        label=r'$f(x)\cdot g(x)$')
ax.fill_between(x_plot, fg_plot, alpha=0.1, color='tab:green')
for gi, c in enumerate(sorted_real_cuts):
    col = gap_colors[gi % len(gap_colors)]
    ax.axvspan(c, c + delta, alpha=0.35, color=col,
               label=f'Cut {gi+1}: [{c:.2f}, {c+delta:.2f}]')
ax.set_xlabel('x', fontsize=12)
ax.set_ylabel(r'$f(x)\cdot g(x)$', fontsize=12)
ax.set_title(r'$f(x)\cdot g(x)$ with cropped-out intervals', fontsize=13)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

fig.suptitle(
    rf'Iterative $g$-cropping: {n_steps} steps, $\delta = {delta}$',
    fontsize=14)
fig.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()