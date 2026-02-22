"""
CO₂-aware computation scheduler.

Uses real power-profile data (f = power in kW) and carbon-intensity
forecasts (g = gCO₂eq/kWh) to find pause intervals of width δ that
minimise total CO₂ emissions.

Approach
--------
1.  Build f(t) from profiled step energies and g(t) from the Electricity
    Maps carbon-intensity API (cached in ci_cache.json).
2.  Iteratively crop out the δ-wide interval of g that reduces
    ∫ f·g the most, shifting g left each time.
3.  After each crop, map the compressed-domain cut position back to
    real (original) time so that the actual pause schedule is expressed
    in wall-clock seconds.
4.  Early-stop when the last *patience* cuts fail to improve CO₂.
5.  Produce four figures:
      (a) f(t) with cut intervals
      (b) g(t) with cut intervals
      (c) f·g  with cut intervals
      (d) CO₂ cost vs. number of highest-impact cutouts applied
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import quad
from scipy.interpolate import interp1d
from scipy.optimize import minimize_scalar

# ────────────────────────────────────────────────────────────────
# Configuration
# ────────────────────────────────────────────────────────────────
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
JSON_PATH = DATA_DIR / "real_data.json"
CI_CACHE_PATH = DATA_DIR / "ci_cache.json"

JOB_DURATION = 8_000.0  # seconds – the job timestamp
DELTA = 10.0  # seconds – width of each pause window
MAX_STEPS = 1_000  # maximum number of cut-out attempts
PATIENCE = 5  # early-stop after this many non-improving cuts
N_INTERP = 8_000  # interpolation grid resolution


# ────────────────────────────────────────────────────────────────
# 1. Build f(t): power in kW from profiled run
# ────────────────────────────────────────────────────────────────


def create_power_function_from_data(data: dict):
    """
    Build f(t) from an already-loaded profiler dict.

    Returns the same triple as :func:`create_power_function`:
      - kw_t(t):  power in kW at time *t* (stepwise)
      - step_times:  cumulative step boundaries (seconds)
      - step_powers: power per step (kW)
    """
    profiled_energy_j = np.array(data["profiled_step_energy_J"])
    time_s = np.array(data["step_time_s"])

    total_steps = len(time_s)
    n_profiled = len(profiled_energy_j)

    repeats = int(np.ceil(total_steps / n_profiled))
    energy_j = np.tile(profiled_energy_j, repeats)[:total_steps]
    power_kw = (energy_j / time_s) / 1000.0

    cumulative_time = np.concatenate(([0.0], np.cumsum(time_s)))

    def kw_t(t):
        idx = np.searchsorted(cumulative_time, t, side="right") - 1
        if isinstance(t, np.ndarray):
            result = np.zeros_like(t, dtype=float)
            valid = (t >= 0) & (t < cumulative_time[-1])
            result[valid] = power_kw[idx[valid]]
            return result
        else:
            if t < 0 or t >= cumulative_time[-1]:
                return 0.0
            return float(power_kw[idx])

    return kw_t, cumulative_time, power_kw


def create_power_function(json_path: str | Path):
    """
    Read the profiler JSON and return:
      - kw_t(t):  power in kW at time *t* (stepwise)
      - step_times:  cumulative step boundaries (seconds)
      - step_powers: power per step (kW)
    """
    with open(json_path, "r") as fh:
        data = json.load(fh)

    return create_power_function_from_data(data)


# ────────────────────────────────────────────────────────────────
# 2. Build g(t): carbon intensity in gCO₂eq/kWh
# ────────────────────────────────────────────────────────────────


def fetch_and_cache_carbon_intensity(cache_path: str | Path, zone: str = "US-TEX-ERCO"):
    """
    Try to load carbon-intensity data from *cache_path*.
    On cache miss, fetch from the Electricity Maps API and save.
    Returns (times_seconds, ci_values).
    """
    cache_path = Path(cache_path)
    if cache_path.exists():
        with open(cache_path, "r") as fh:
            cached = json.load(fh)
        return np.array(cached["times_seconds"]), np.array(cached["ci_values"])

    # ── Live fetch ─────────────────────────────────────────────
    from datetime import datetime

    import requests

    TOKEN = "4N4rpoLvWqfvQ0yJdHN0"
    BASE = "https://api.electricitymaps.com/v3/carbon-intensity"
    HEADERS = {"auth-token": TOKEN}

    def _fetch(endpoint):
        r = requests.get(
            f"{BASE}/{endpoint}",
            params={"zone": zone, "temporalGranularity": "5_minutes"},
            headers=HEADERS,
        )
        r.raise_for_status()
        return r.json()[endpoint]

    history = _fetch("history")
    forecast = _fetch("forecast")

    def _parse(s):
        return datetime.fromisoformat(s.replace("Z", "+00:00"))

    timeline: dict = {}
    for e in history:
        timeline[_parse(e["datetime"])] = e["carbonIntensity"]
    for e in forecast:
        dt = _parse(e["datetime"])
        if dt not in timeline:
            timeline[dt] = e["carbonIntensity"]

    sorted_dts = sorted(timeline.keys())
    t0 = sorted_dts[0]
    times_sec = [(dt - t0).total_seconds() for dt in sorted_dts]
    ci_vals = [timeline[dt] for dt in sorted_dts]

    with open(cache_path, "w") as fh:
        json.dump({"times_seconds": times_sec, "ci_values": ci_vals}, fh)

    return np.array(times_sec), np.array(ci_vals)


def create_co2_interpolator(times_seconds, ci_values):
    """Return a function g(t) → gCO₂eq/kWh at time *t* (seconds)."""

    def g(t):
        return np.interp(t, times_seconds, ci_values)

    return g


# ────────────────────────────────────────────────────────────────
# 3. g-cropping optimisation machinery
# ────────────────────────────────────────────────────────────────


def J_crop(a, f_func, g_func, T_g, delta):
    """
    Objective: ∫₀^{T_g - δ} f(x) · g'(x) dx
    where g' crops [a, a+δ] out of g and shifts left.
    """
    T_new = T_g - delta
    if a < 0 or a > T_new:
        return 1e30

    def integrand(x):
        if x < a:
            return float(f_func(x)) * float(g_func(x))
        else:
            return float(f_func(x)) * float(g_func(x + delta))

    val, _ = quad(integrand, 0, T_new, limit=400, points=[a])
    return val


def compressed_to_real(x, sorted_real_cuts, delta):
    """Map compressed-space position → original g coordinate."""
    pos = x
    for c in sorted_real_cuts:
        if pos >= c:
            pos += delta
        else:
            break
    return pos


# ────────────────────────────────────────────────────────────────
# 4. Main optimisation loop
# ────────────────────────────────────────────────────────────────


def run_optimisation(
    f_func,
    g_func,
    T: float,
    delta: float = DELTA,
    max_steps: int = MAX_STEPS,
    patience: int = PATIENCE,
    n_interp: int = N_INTERP,
    verbose: bool = True,
):
    """
    Iteratively crop δ-wide windows from g to minimise ∫ f·g.

    Returns
    -------
    sorted_real_cuts : list[float]
        Start positions (original g-space) of accepted pause windows.
    J_history : list[float]
        Objective value after each accepted cut.
    cut_savings : list[float]
        CO₂ saving (relative to no-cut baseline) of each accepted cut,
        sorted by descending impact.
    baseline_J : float
        Original ∫ f·g with no pauses.
    """
    # ── Build working copy of g on a grid ──────────────────────
    cur_x = np.linspace(0, T, n_interp)
    cur_g_vals = g_func(cur_x)
    cur_g = interp1d(
        cur_x, cur_g_vals, kind="linear", bounds_error=False, fill_value=0.0
    )

    baseline_J = quad(lambda x: float(f_func(x)) * float(g_func(x)), 0, T, limit=400)[0]

    T_g = T
    sorted_real_cuts = []
    J_history = []
    no_improve_count = 0
    best_J = baseline_J
    best_real_cuts = []  # best schedule seen so far
    prev_J = baseline_J

    # Piece tracking: intervals of the compressed domain where g is
    # still present.  After each accepted cut at a_s the piece
    # containing a_s splits; later pieces shift left by δ.
    pieces = [(0.0, T)]

    if verbose:
        print(f"Baseline CO₂ cost (∫ f·g): {baseline_J:.4f}")
        print(f"δ = {delta} s, max steps = {max_steps}, patience = {patience}")
        print("─" * 65)

    for step_i in range(max_steps):
        T_new = T_g - delta
        if T_new <= 0:
            if verbose:
                print(f"  Step {step_i + 1}: domain exhausted, stopping.")
            break

        _g, _T_g = cur_g, T_g

        def J_this(a, _f=f_func, _g=_g, _T_g=_T_g, _delta=delta):
            return J_crop(a, _f, _g, _T_g, _delta)

        # ── Per-piece local optimisation ──────────────────────
        best_a = None
        best_J_piece = 1e30
        best_pi = None

        for pi, (p_lo, p_hi) in enumerate(pieces):
            width = p_hi - p_lo
            if width < 1e-10:
                continue
            # Clamp bounds to [0, T_new]
            lo = max(0.0, p_lo)
            hi = min(T_new, p_hi)
            if hi - lo < 1e-10:
                continue
            try:
                res_loc = minimize_scalar(
                    J_this,
                    bounds=(lo, hi),
                    method="bounded",
                    options={"xatol": 1e-10, "maxiter": 2000},
                )
                if res_loc.fun < best_J_piece:
                    best_J_piece = res_loc.fun
                    best_a = res_loc.x
                    best_pi = pi
            except Exception:
                pass

        if best_a is None:
            if verbose:
                print(f"  Step {step_i + 1}: no valid piece, stopping.")
            break

        J_val = best_J_piece

        # Always accept the cut, but track the best schedule
        a_s = best_a

        # Map to original g-space
        real_a = compressed_to_real(a_s, sorted_real_cuts, delta)
        sorted_real_cuts.append(real_a)
        sorted_real_cuts.sort()

        # ── Update piece list ─────────────────────────────────
        p_lo, p_hi = pieces[best_pi]
        new_pieces = []
        for pi, (lo_p, hi_p) in enumerate(pieces):
            if pi < best_pi:
                new_pieces.append((lo_p, hi_p))
            elif pi == best_pi:
                if a_s - p_lo > 1e-10:
                    new_pieces.append((p_lo, a_s))
                if p_hi - a_s - delta > 1e-10:
                    new_pieces.append((a_s, p_hi - delta))
            else:
                new_pieces.append((lo_p - delta, hi_p - delta))
        pieces = [(max(0, lo), hi) for lo, hi in new_pieces if hi > lo + 1e-10]

        J_history.append(J_val)
        prev_J = J_val

        # Track the best schedule
        if J_val < best_J - 1e-12:
            best_J = J_val
            best_real_cuts = list(sorted_real_cuts)
            no_improve_count = 0
        else:
            no_improve_count += 1

        if verbose and (
            step_i < 10
            or (step_i + 1) % 50 == 0
            or step_i == max_steps - 1
            or no_improve_count > 0
        ):
            tag = (
                ""
                if no_improve_count == 0
                else f" [no improve {no_improve_count}/{patience}]"
            )
            print(
                f"  Step {step_i + 1}: a_real={real_a:.1f} s, "
                f"J={J_val:.4f}, saved {baseline_J - J_val:.4f} "
                f"({(baseline_J - J_val) / baseline_J * 100:.2f}%){tag}"
            )

        if no_improve_count >= patience:
            if verbose:
                print(
                    f"  Early stopping at step {step_i + 1}: "
                    f"patience exhausted. "
                    f"Reverting to best schedule "
                    f"({len(best_real_cuts)} cuts, "
                    f"J={best_J:.4f})."
                )
            break

        # Rebuild cropped g
        new_x = np.linspace(0, T_new, n_interp)
        new_g_vals = np.zeros(n_interp)
        for k, xv in enumerate(new_x):
            if xv < a_s:
                new_g_vals[k] = float(cur_g(xv))
            else:
                new_g_vals[k] = float(cur_g(xv + delta))

        cur_g = interp1d(
            new_x, new_g_vals, kind="linear", bounds_error=False, fill_value=0.0
        )
        T_g = T_new

    # Use the best tracked schedule
    sorted_real_cuts = best_real_cuts if best_real_cuts else sorted_real_cuts
    final_J = best_J

    # ── Per-cut savings (sorted by descending impact) ──────────
    # Compute the marginal saving of each cut by replaying one at a time
    # on the original g, ranked by single-cut impact.
    cut_savings = []
    for c in sorted_real_cuts:
        # J with only this single cut
        def _make_integrand(cut_pos):
            def _integrand(x):
                if x < cut_pos:
                    return float(f_func(x)) * float(g_func(x))
                else:
                    return float(f_func(x)) * float(g_func(x + delta))

            return _integrand

        T_single = T - delta
        j_single = quad(_make_integrand(c), 0, T_single, limit=400, points=[c])[0]
        cut_savings.append(baseline_J - j_single)

    # Sort descending
    order = np.argsort(cut_savings)[::-1]
    cut_savings = [cut_savings[i] for i in order]
    ordered_cuts = [sorted_real_cuts[i] for i in order]

    if verbose:
        print("─" * 65)
        print(f"Best schedule: {len(sorted_real_cuts)} cuts")
        print(
            f"Best CO₂ cost: {final_J:.4f}  "
            f"(saved {baseline_J - final_J:.4f}, "
            f"{(baseline_J - final_J) / baseline_J * 100:.2f}%)"
        )

    return sorted_real_cuts, J_history, cut_savings, ordered_cuts, baseline_J


# ────────────────────────────────────────────────────────────────
# 5. Plotting
# ────────────────────────────────────────────────────────────────


def plot_results(
    f_func,
    g_func,
    T,
    delta,
    sorted_real_cuts,
    J_history,
    cut_savings,
    ordered_cuts,
    baseline_J,
):
    """Produce four figures."""

    x_plot = np.linspace(0, T, 4000)
    f_plot = np.array([float(f_func(x)) for x in x_plot])
    g_plot = g_func(x_plot)
    fg_plot = f_plot * g_plot

    gap_colors = plt.cm.tab10.colors

    # ── Figure 1: three stacked subplots (f, g, f·g with cuts) ──
    fig, axes = plt.subplots(3, 1, figsize=(16, 13), sharex=True)

    for ax_i, (ydata, ylabel, title, color) in enumerate(
        [
            (
                f_plot,
                "Power  f(t)  [kW]",
                "Power profile f(t) with pause intervals",
                "tab:blue",
            ),
            (
                g_plot,
                r"Carbon intensity  g(t)  [gCO$_2$eq/kWh]",
                "Carbon intensity g(t) with pause intervals",
                "tab:orange",
            ),
            (
                fg_plot,
                r"f(t)$\cdot$g(t)  [gCO$_2$eq·s/kWh·s → gCO$_2$eq]",
                r"Product f(t)$\cdot$g(t) with pause intervals",
                "tab:green",
            ),
        ]
    ):
        ax = axes[ax_i]
        if ax_i == 1:  # bar plot for g
            bar_width = x_plot[1] - x_plot[0]
            ax.bar(x_plot, ydata, width=bar_width, color=color, alpha=0.7, linewidth=0)
        else:
            ax.plot(x_plot, ydata, color=color, linewidth=1.2)
        if ax_i == 2:
            ax.fill_between(x_plot, ydata, alpha=0.08, color=color)
        for gi, c in enumerate(sorted_real_cuts):
            col = gap_colors[gi % len(gap_colors)]
            kw = dict(alpha=0.30, color=col)
            if gi < 8:  # legend only for first few
                kw["label"] = f"Pause {gi + 1}: [{c:.0f}, {c + delta:.0f}] s"
            ax.axvspan(c, c + delta, **kw)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=12)
        if ax_i == 0:
            ax.legend(fontsize=8, ncol=2, loc="upper right")
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time  t  [s]", fontsize=11)
    fig.suptitle(
        f"CO₂-aware pause schedule: {len(sorted_real_cuts)} pauses "
        f"of δ = {delta:.0f} s",
        fontsize=14,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    # ── Figure 2: cumulative CO₂ saved vs. # of top-impact cuts ──
    fig2, ax2 = plt.subplots(figsize=(10, 5))

    cum_savings = np.cumsum(cut_savings)
    n_cuts_range = np.arange(1, len(cut_savings) + 1)

    ax2.plot(
        n_cuts_range,
        baseline_J - cum_savings,
        "o-",
        markersize=3,
        color="tab:red",
        label="CO₂ cost",
    )
    ax2.axhline(
        baseline_J,
        color="grey",
        linestyle=":",
        alpha=0.7,
        label=f"No pauses: {baseline_J:.1f}",
    )
    ax2.set_xlabel("Number of highest-impact pauses applied", fontsize=11)
    ax2.set_ylabel(r"Total CO$_2$ cost  [gCO$_2$eq·s/kWh]", fontsize=11)
    ax2.set_title("CO₂ cost vs. number of top-impact pause intervals", fontsize=13)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    fig2.tight_layout()

    plt.show()


# ────────────────────────────────────────────────────────────────
# 6. Main entry point
# ────────────────────────────────────────────────────────────────


def main():
    print("═" * 65)
    print("  CO₂-aware computation scheduler")
    print("═" * 65)

    # ── Power profile f(t) ─────────────────────────────────────
    print("\n[1] Loading power profile …")
    f_func, step_boundaries, step_powers = create_power_function(JSON_PATH)
    run_duration = step_boundaries[-1]
    T = min(JOB_DURATION, run_duration)
    print(f"    Profiled run duration : {run_duration:.0f} s")
    print(f"    Using job duration T  : {T:.0f} s")

    # ── Carbon intensity g(t) ──────────────────────────────────
    print("\n[2] Loading carbon intensity …")
    ci_times, ci_values = fetch_and_cache_carbon_intensity(CI_CACHE_PATH)
    g_func = create_co2_interpolator(ci_times, ci_values)
    print(f"    Timeline span: 0 – {ci_times[-1]:.0f} s  ({len(ci_times)} data points)")
    print(
        f"    CI range: {ci_values.min():.0f} – {ci_values.max():.0f} "
        f"gCO₂eq/kWh  (mean {ci_values.mean():.0f})"
    )

    # ── Optimise ───────────────────────────────────────────────
    print(f"\n[3] Running iterative g-cropping optimisation …\n")
    sorted_real_cuts, J_history, cut_savings, ordered_cuts, baseline_J = (
        run_optimisation(
            f_func, g_func, T, delta=DELTA, max_steps=MAX_STEPS, patience=PATIENCE
        )
    )

    # ── Plot ───────────────────────────────────────────────────
    print("\n[4] Plotting results …")
    plot_results(
        f_func,
        g_func,
        T,
        DELTA,
        sorted_real_cuts,
        J_history,
        cut_savings,
        ordered_cuts,
        baseline_J,
    )


if __name__ == "__main__":
    main()
