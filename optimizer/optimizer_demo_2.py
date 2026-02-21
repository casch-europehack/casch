import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# ── 1. Breakpoints and Stepwise Setup ────────────────────────────────────
T = 10.0
delta = 1.0

# Define the discrete set of breakpoints
N_BREAKPOINTS = 31
breakpoints = np.linspace(0, T, N_BREAKPOINTS)


# Original continuous functions
def f_orig(t):
    return np.sin(1.5 * t) + 2.0


def g_orig(t):
    return np.exp(-0.05 * t)


# Evaluate original functions at breakpoints to freeze their plateau values
f_vals = f_orig(breakpoints)
g_vals = g_orig(breakpoints)


def f_step(t):
    """Stepwise f(t) evaluated precisely at breakpoints."""
    idx = np.searchsorted(breakpoints, t, side="right") - 1
    idx = np.clip(idx, 0, len(breakpoints) - 1)
    return f_vals[idx] if np.isscalar(t) else f_vals[idx.astype(int)]


def g_step(t):
    """Stepwise g(t) evaluated precisely at breakpoints."""
    idx = np.searchsorted(breakpoints, t, side="right") - 1
    idx = np.clip(idx, 0, len(breakpoints) - 1)
    return g_vals[idx] if np.isscalar(t) else g_vals[idx.astype(int)]


# ── 2. Vectorized Mappings ───────────────────────────────────────────────
def u_map(t, a, b):
    """Vectorized mapping of integration time 't' to the original domain 'u'."""
    return np.where(
        t < a,
        t,
        np.where(
            t < a + b + delta,
            (t - a) * (b / (b + delta) if b > 1e-8 else 0.0) + a,
            t - delta,
        ),
    )


def scale_map(t, a, b):
    """Vectorized integration scaling factor (dt derivative)."""
    if b <= 1e-8:  # Handle collapsed interval gracefully
        return np.where((t >= a) & (t <= a + delta), 0.0, 1.0)
    else:
        return np.where((t >= a) & (t <= a + b + delta), b / (b + delta), 1.0)


# ── 3. Exact Integration via Dense Riemann Sum ───────────────────────────
# Because stepwise functions confuse scipy.integrate.quad, a dense Riemann
# sum is numerically stable, exact for step functions, and ~100x faster.
t_dense = np.linspace(0, T + delta, 50000)
dt = t_dense[1] - t_dense[0]


def objective(a, b):
    # Map t to u(t) and get the scale factor
    u_dense = u_map(t_dense, a, b)
    s_dense = scale_map(t_dense, a, b)

    # Evaluate stepwise functions natively across arrays
    integrand = f_step(u_dense) * g_step(t_dense) * s_dense
    return np.sum(integrand) * dt


# ── 4. Grid Search Optimization ──────────────────────────────────────────
# Constrain choices for a and b strictly to the breakpoints
valid_pairs = [(a, b) for a in breakpoints for b in breakpoints if a + b <= T + 1e-8]

best_J = np.inf
best_a, best_b = None, None

# Execute Brute-force Search with a progress bar
pbar = tqdm(
    valid_pairs,
    desc="Grid Search (a, b)",
    bar_format="{l_bar}{bar} [ time left: {remaining} ]",
)
for a_cand, b_cand in pbar:
    j_val = objective(a_cand, b_cand)
    if j_val < best_J:
        best_J = j_val
        best_a, best_b = a_cand, b_cand
pbar.close()

print("-" * 30)
print(f"Optimal a* : {best_a:.4f}")
print(f"Optimal b* : {best_b:.4f}")
print(f"Minimum J* : {best_J:.4f}")
