import numpy as np
from scipy import integrate, optimize


# ── Define your functions and parameters ───────────────────────────────
def f(t):
    return np.sin(0.5 * t) + 1.5


def g(t):
    return np.exp(-0.1 * t)


T, delta = 10.0, 1.0


def h(t):
    return f(t) * g(t)


def H(x):
    return integrate.quad(h, 0, x)[0]  # cumulative integral


H_total = H(T + delta)  # constant term; precompute once


# ── Objective and exact gradient ────────────────────────────────────────
def objective(p):
    a, b = p
    return H_total - (delta / (b + delta)) * (H(a + b + delta) - H(a))


def gradient(p):
    a, b = p
    window = H(a + b + delta) - H(a)
    dJda = -(delta / (b + delta)) * (h(a + b + delta) - h(a))
    dJdb = (delta / (b + delta) ** 2) * window - (delta / (b + delta)) * h(
        a + b + delta
    )
    return np.array([dJda, dJdb])


# ── Feasible set: a,b ≥ 0  and  a+b ≤ T ────────────────────────────────
bounds = [(0, T), (0, T)]
constraints = [{"type": "ineq", "fun": lambda p: T - p[0] - p[1]}]

# ── Multi-start SLSQP (handles non-convexity) ───────────────────────────
rng, best = np.random.default_rng(42), None
for _ in range(200):
    a0 = rng.uniform(0, T)
    b0 = rng.uniform(0, T - a0)
    res = optimize.minimize(
        objective,
        [a0, b0],
        jac=gradient,
        bounds=bounds,
        constraints=constraints,
        method="SLSQP",
        options={"ftol": 1e-12},
    )
    if res.success and (best is None or res.fun < best.fun):
        best = res

print(f"a* = {best.x[0]:.4f},  b* = {best.x[1]:.4f},  J* = {best.fun:.6f}")
