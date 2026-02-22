"""
Unit tests for the gap-insertion optimizer (optimizer_demo.py).
"""
import numpy as np
import pytest
from scipy.interpolate import interp1d
from scipy.integrate import quad
from scipy.optimize import minimize_scalar


# ── Import the core function from the demo module ───────────
# f_modified lives in optimizer_demo but that file executes top-level code,
# so we redefine the pure functions here to keep the tests side-effect-free.


def f_modified(x, a, delta, f_func, T_cur):
    """Insert a zero-gap of width *delta* at position *a* into *f_func*.

    Returns the value of the modified function f' at x:
        [0, a)            → f(x)
        [a, a+delta]      → 0
        (a+delta, T+delta] → f(x - delta)
    """
    if x < 0 or x > T_cur + delta:
        return 0.0
    if x < a:
        return float(f_func(x))
    elif x <= a + delta:
        return 0.0
    else:
        return float(f_func(x - delta))


def make_objective(f_func, g_func, T, delta):
    """Return J(a) = ∫₀^{T+δ} f'(x)·g(x) dx  for a given gap position a."""

    def J(a):
        if a < 0 or a > T:
            return 1e12

        def integrand(x):
            return f_modified(x, a, delta, f_func, T) * float(g_func(x))

        return quad(integrand, 0, T + delta, limit=200,
                    points=[a, a + delta])[0]

    return J


def optimize_gap(f_func, g_func, T, delta):
    """Find the gap position a* that minimizes J(a)."""
    J = make_objective(f_func, g_func, T, delta)
    res = minimize_scalar(J, bounds=(0.0, T), method='bounded',
                          options={'xatol': 1e-10, 'maxiter': 1000})
    return res


# ═══════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════

def _const_interp(c, x_lo=0.0, x_hi=20.0, n=2000):
    """Return an interp1d that is constantly *c* on [x_lo, x_hi]."""
    xs = np.linspace(x_lo, x_hi, n)
    return interp1d(xs, np.full_like(xs, c), kind='linear',
                    bounds_error=False, fill_value=0.0)


def _gaussian(x, mu, sigma, amp=1.0):
    return amp * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def _make_three_peak_g(centers, amplitudes, sigma, x_hi=20.0, n=4000):
    """Return an interp1d for g with three Gaussian peaks."""
    xs = np.linspace(0, x_hi, n)
    ys = np.zeros_like(xs)
    for mu, amp in zip(centers, amplitudes):
        ys += _gaussian(xs, mu, amp=amp, sigma=sigma)
    return interp1d(xs, ys, kind='cubic', bounds_error=False, fill_value=0.0)


# ═══════════════════════════════════════════════════════════════
#  Tests for f_modified
# ═══════════════════════════════════════════════════════════════

class TestFModified:
    """Tests for the gap-insertion helper f_modified."""

    def test_identity_before_gap(self):
        """For x < a, f_modified should equal f(x)."""
        f = _const_interp(3.0)
        a, delta, T = 5.0, 1.0, 10.0
        for x in [0.0, 1.0, 4.99]:
            assert f_modified(x, a, delta, f, T) == pytest.approx(3.0)

    def test_zero_in_gap(self):
        """Inside the gap [a, a+delta], f_modified should be 0."""
        f = _const_interp(3.0)
        a, delta, T = 5.0, 1.0, 10.0
        for x in [5.0, 5.5, 6.0]:
            assert f_modified(x, a, delta, f, T) == 0.0

    def test_shifted_after_gap(self):
        """For x > a + delta, f_modified(x) should equal f(x - delta)."""
        xs = np.linspace(0, 15, 3000)
        f_vals = np.sin(xs) + 2.0
        f = interp1d(xs, f_vals, kind='cubic', bounds_error=False, fill_value=0.0)
        a, delta, T = 4.0, 1.5, 10.0
        for x in [a + delta + 0.01, 7.0, T + delta]:
            expected = float(f(x - delta))
            assert f_modified(x, a, delta, f, T) == pytest.approx(expected, abs=1e-8)

    def test_zero_outside_domain(self):
        """f_modified should return 0 for x < 0 or x > T + delta."""
        f = _const_interp(5.0)
        a, delta, T = 3.0, 2.0, 10.0
        assert f_modified(-0.1, a, delta, f, T) == 0.0
        assert f_modified(T + delta + 0.1, a, delta, f, T) == 0.0

    def test_area_preserved(self):
        """Inserting a gap should not change the total area of f
        (gap is 0 and the rest is just f shifted)."""
        xs = np.linspace(0, 15, 3000)
        f_vals = 0.6 + 0.25 * np.sin(1.5 * xs)
        f = interp1d(xs, f_vals, kind='cubic', bounds_error=False, fill_value=0.0)
        T, delta, a = 10.0, 1.5, 3.7

        orig_area = quad(lambda x: float(f(x)), 0, T)[0]
        mod_area = quad(lambda x: f_modified(x, a, delta, f, T), 0, T + delta)[0]
        assert mod_area == pytest.approx(orig_area, rel=1e-6)

    def test_gap_at_zero(self):
        """When a = 0 the gap occupies [0, delta] and f is simply shifted."""
        xs = np.linspace(0, 15, 3000)
        f_vals = xs + 1.0  # linear
        f = interp1d(xs, f_vals, kind='linear', bounds_error=False, fill_value=0.0)
        T, delta = 10.0, 2.0
        a = 0.0
        # In gap
        assert f_modified(0.5, a, delta, f, T) == 0.0
        # After gap: f_modified(x) = f(x - delta)
        assert f_modified(5.0, a, delta, f, T) == pytest.approx(float(f(3.0)), abs=1e-8)

    def test_gap_at_T(self):
        """When a = T the gap occupies [T, T+delta]; f is unchanged on [0,T)."""
        f = _const_interp(2.0)
        T, delta = 10.0, 1.5
        a = T
        assert f_modified(5.0, a, delta, f, T) == pytest.approx(2.0)
        assert f_modified(T + 0.5, a, delta, f, T) == 0.0


# ═══════════════════════════════════════════════════════════════
#  Tests for the objective J(a)
# ═══════════════════════════════════════════════════════════════

class TestObjective:
    """Tests for the integral objective J(a)."""

    def test_out_of_bounds_penalty(self):
        """J(a) must return a large penalty for a outside [0, T]."""
        f = _const_interp(1.0)
        g = _const_interp(1.0)
        J = make_objective(f, g, T=10.0, delta=1.0)
        assert J(-0.1) == 1e12
        assert J(10.1) == 1e12

    def test_constant_f_constant_g(self):
        """With f ≡ c₁ and g ≡ c₂, the gap zeroes out a segment of width δ,
        so J(a) = c₁·c₂·T  (same for all a), because the gap removes δ
        but the domain extends by δ."""
        c1, c2, T, delta = 2.0, 3.0, 10.0, 1.5
        f = _const_interp(c1, x_hi=T + delta + 2)
        g = _const_interp(c2, x_hi=T + delta + 2)
        J = make_objective(f, g, T, delta)
        expected = c1 * c2 * T  # gap removes δ, domain extends by δ → net area = c1*T
        assert J(0.0) == pytest.approx(expected, rel=1e-6)
        assert J(5.0) == pytest.approx(expected, rel=1e-6)
        assert J(T) == pytest.approx(expected, rel=1e-6)

    def test_symmetry(self):
        """When f and g are both symmetric about T/2, J(a) = J(T - a)."""
        T, delta = 8.0, 0.5
        xs = np.linspace(0, T + delta + 2, 4000)
        mid = T / 2.0
        f_vals = 1.0 + np.cos(2 * np.pi * (xs - mid) / T)
        g_vals = 1.0 + np.cos(2 * np.pi * (xs - mid) / T)
        f = interp1d(xs, f_vals, kind='cubic', bounds_error=False, fill_value=0.0)
        g = interp1d(xs, g_vals, kind='cubic', bounds_error=False, fill_value=0.0)
        J = make_objective(f, g, T, delta)
        # Check a few symmetric pairs
        for a in [0.5, 1.5, 3.0]:
            assert J(a) == pytest.approx(J(T - a), rel=1e-4)


# ═══════════════════════════════════════════════════════════════
#  Tests for the optimizer
# ═══════════════════════════════════════════════════════════════

class TestOptimizer:
    """Tests for the full gap-position optimizer."""

    def test_constant_f_three_peak_g(self):
        """f ≡ 1, g has 3 Gaussian peaks of width ≈ δ.

        With constant f the objective simplifies to
            J(a) = ∫₀^{T+δ} g(x) dx  −  ∫_a^{a+δ} g(x) dx
        so minimising J is the same as *maximising* the integral of g
        over a window of width δ.  The gap should therefore land on
        the tallest peak.
        """
        T = 10.0
        delta = 1.0
        sigma = delta / (2 * 2.35)  # FWHM ≈ delta  (2.35 ≈ 2√(2 ln2))
        centers = [2.5, 5.0, 7.5]
        amplitudes = [1.0, 3.0, 2.0]  # tallest peak at 5.0

        f = _const_interp(1.0, x_hi=T + delta + 2)
        g = _make_three_peak_g(centers, amplitudes, sigma, x_hi=T + delta + 2)

        res = optimize_gap(f, g, T, delta)
        a_opt = res.x

        # The gap [a*, a*+δ] should be centred on the tallest peak (5.0)
        gap_center = a_opt + delta / 2.0
        assert gap_center == pytest.approx(5.0, abs=0.15), \
            f"Expected gap centred ≈ 5.0, got {gap_center:.4f}"

        # J(a*) should be less than placing the gap on either other peak
        J = make_objective(f, g, T, delta)
        assert res.fun < J(2.5 - delta / 2)
        assert res.fun < J(7.5 - delta / 2)

    def test_gap_avoids_high_g_region(self):
        """When g is large in one area the gap should land there to
        minimise the integral (for constant f)."""
        T = 10.0
        delta = 1.0
        xs = np.linspace(0, T + delta + 2, 4000)
        # g is a step function: high in [3, 4], low elsewhere
        g_vals = np.where((xs >= 3.0) & (xs <= 4.0), 10.0, 1.0)
        g = interp1d(xs, g_vals, kind='linear', bounds_error=False, fill_value=1.0)
        f = _const_interp(1.0, x_hi=T + delta + 2)

        res = optimize_gap(f, g, T, delta)
        a_opt = res.x
        # Gap should cover the high-g region [3, 4]
        assert a_opt == pytest.approx(3.0, abs=0.2)

    def test_optimizer_finds_global_min(self):
        """J(a*) must be ≤ J(a) for a dense sample of a values."""
        T, delta = 10.0, 1.5
        xs = np.linspace(0, T + delta + 2, 4000)
        f_vals = 0.6 + 0.25 * np.sin(1.5 * xs + 0.3) + 0.2 * np.cos(3.2 * xs + 1.0)
        g_vals = 0.8 + 0.3 * np.sin(1.1 * xs + 0.5) + 0.2 * np.cos(2.7 * xs + 1.5)
        f = interp1d(xs, f_vals, kind='cubic', bounds_error=False, fill_value=0.0)
        g = interp1d(xs, g_vals, kind='cubic', bounds_error=False, fill_value="extrapolate")

        res = optimize_gap(f, g, T, delta)
        J = make_objective(f, g, T, delta)
        for a_test in np.linspace(0, T, 200):
            assert res.fun <= J(a_test) + 1e-8

    def test_area_preserved_after_optimization(self):
        """Total area under f should be the same before and after
        gap insertion at the optimised position."""
        T, delta = 10.0, 1.0
        xs = np.linspace(0, T + delta + 2, 4000)
        f_vals = 2.0 + np.sin(xs)
        f = interp1d(xs, f_vals, kind='cubic', bounds_error=False, fill_value=0.0)
        g = _const_interp(1.0, x_hi=T + delta + 2)

        res = optimize_gap(f, g, T, delta)
        a_opt = res.x

        orig_area = quad(lambda x: float(f(x)), 0, T)[0]
        mod_area = quad(lambda x: f_modified(x, a_opt, delta, f, T), 0, T + delta)[0]
        assert mod_area == pytest.approx(orig_area, rel=1e-6)


# ═══════════════════════════════════════════════════════════════
#  Tests for the iterative (multi-step) sweep
# ═══════════════════════════════════════════════════════════════

class TestIterativeSweep:
    """Tests for applying the gap insertion repeatedly with small steps."""

    @staticmethod
    def _run_iterative_sweep(f_func, g_func, T, d_step, n_steps):
        """Replicate the iterative sweep from the demo script."""
        n_interp = 4000
        cur_x = np.linspace(0, T, n_interp)
        cur_f_vals = np.array([float(f_func(x)) for x in cur_x])
        cur_f = interp1d(cur_x, cur_f_vals, kind='linear',
                         bounds_error=False, fill_value=0.0)
        current_T = T
        results = []

        for _ in range(n_steps):
            d = d_step
            T_cur = current_T

            def J_step(a, _d=d, _T=T_cur, _cur_f=cur_f):
                if a < 0 or a > _T:
                    return 1e12

                def integrand(x):
                    if x < a:
                        return float(_cur_f(x)) * float(g_func(x))
                    elif x <= a + _d:
                        return 0.0
                    else:
                        return float(_cur_f(x - _d)) * float(g_func(x))

                return quad(integrand, 0, _T + _d, limit=200,
                            points=[a, a + _d])[0]

            res = minimize_scalar(J_step, bounds=(0.0, T_cur), method='bounded',
                                  options={'xatol': 1e-10, 'maxiter': 1000})
            a_s = res.x
            current_T_new = T_cur + d

            new_x = np.linspace(0, current_T_new, n_interp)
            new_f_vals = np.zeros(n_interp)
            for k, xv in enumerate(new_x):
                if xv < 0 or xv > current_T_new:
                    new_f_vals[k] = 0.0
                elif xv < a_s:
                    new_f_vals[k] = float(cur_f(xv))
                elif xv <= a_s + d:
                    new_f_vals[k] = 0.0
                else:
                    new_f_vals[k] = float(cur_f(xv - d))

            cur_f = interp1d(new_x, new_f_vals, kind='linear',
                             bounds_error=False, fill_value=0.0)
            current_T = current_T_new
            results.append({'a': a_s, 'J': res.fun, 'T': current_T})

        return results, cur_f, current_T

    def test_monotone_decrease(self):
        """Each iterative step should yield J* ≤ previous J* (or stay equal),
        because we are extending the domain and zeroing out more of f."""
        T, d_step, n_steps = 6.0, 0.1, 10
        xs = np.linspace(0, T + n_steps * d_step + 2, 4000)
        f_vals = 1.0 + 0.5 * np.sin(2.0 * xs)
        g_vals = 1.0 + 0.4 * np.cos(1.3 * xs)
        f = interp1d(xs, f_vals, kind='cubic', bounds_error=False, fill_value=0.0)
        g = interp1d(xs, g_vals, kind='cubic', bounds_error=False, fill_value="extrapolate")

        results, _, _ = self._run_iterative_sweep(f, g, T, d_step, n_steps)
        J_vals = [r['J'] for r in results]
        for i in range(1, len(J_vals)):
            assert J_vals[i] <= J_vals[i - 1] + 1e-8, \
                f"Step {i}: J increased from {J_vals[i-1]:.6f} to {J_vals[i]:.6f}"

    def test_area_preserved_iterative(self):
        """After all iterative steps, total area of f should equal the original."""
        T, d_step, n_steps = 6.0, 0.1, 8
        xs = np.linspace(0, T + n_steps * d_step + 2, 4000)
        f_vals = 2.0 + np.sin(xs)
        f = interp1d(xs, f_vals, kind='cubic', bounds_error=False, fill_value=0.0)
        g = _const_interp(1.0, x_hi=T + n_steps * d_step + 2)

        orig_area = quad(lambda x: float(f(x)), 0, T)[0]
        _, cur_f, current_T = self._run_iterative_sweep(f, g, T, d_step, n_steps)
        mod_area = quad(lambda x: float(cur_f(x)), 0, current_T)[0]
        # Linear interpolation in the sweep introduces small errors
        assert mod_area == pytest.approx(orig_area, rel=1e-3)
