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

# ────────────────────────────────────────────────────────────────
# Configuration
# ────────────────────────────────────────────────────────────────
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
JSON_PATH = DATA_DIR / "real_data.json"
CI_CACHE_PATH = DATA_DIR / "ci_cache.json"

JOB_DURATION  = 8_000.0      # seconds – the job timestamp
DELTA         = 10.0          # seconds – width of each pause window
MAX_STEPS     = 1_000         # maximum number of cut-out attempts
PATIENCE      = 50             # early-stop after this many non-improving cuts
MIN_IMPROVE   = 1e-6          # stop when marginal improvement < this fraction
N_INTERP      = 8_000         # interpolation grid resolution


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


def compressed_to_real(x, sorted_real_cuts, delta):
    """Map compressed-space position → original g coordinate."""
    pos = x
    for c in sorted_real_cuts:
        if pos >= c:
            pos += delta
        else:
            break
    return pos


def find_best_cut(f_arr, g_arr, h, delta_idx, margin=0):
    """
    Given fixed f and g arrays on a uniform grid with spacing *h*,
    find the cut position that minimises ∫₀ᵀ f·g' where g' is g
    with [i, i+delta_idx) removed and the right part shifted left.

    The integral is always over [0, T] (the domain of f, length N).
    g must be at least N + delta_idx long so that shifted values
    from beyond T can slide in.

    Parameters
    ----------
    f_arr     : ndarray, shape (N,)    – power profile on [0, T]
    g_arr     : ndarray, shape (≥ N + delta_idx,)  – carbon intensity
    h         : float                  – grid spacing
    delta_idx : int                    – width of cut in grid cells
    margin    : int                    – exclude the last *margin*
                indices from the search (avoids trivial boundary cuts)

    Returns
    -------
    best_idx : int     – grid index of the optimal cut start
    best_J   : float   – objective value after this cut
    """
    N = len(f_arr)
    assert len(g_arr) >= N + delta_idx, (
        f"g too short: need {N + delta_idx}, have {len(g_arr)}"
    )

    # f·g (no shift) and f·g_shifted (cut applied), both length N
    fg = f_arr * g_arr[:N]
    fg_shift = f_arr * g_arr[delta_idx : delta_idx + N]

    # Cumulative trapezoidal sums (length N)
    fg_pan = 0.5 * h * (fg[:-1] + fg[1:])
    gs_pan = 0.5 * h * (fg_shift[:-1] + fg_shift[1:])

    F_cum = np.empty(N)
    G_cum = np.empty(N)
    F_cum[0] = 0.0
    G_cum[0] = 0.0
    np.cumsum(fg_pan, out=F_cum[1:])
    np.cumsum(gs_pan, out=G_cum[1:])

    # J(i) = ∫₀^{x_i} f·g  +  ∫_{x_i}^T f·g_shifted
    J_all = F_cum + (G_cum[-1] - G_cum)

    # Restrict search to [0, N - margin) to exclude trivial boundary cuts
    search_end = max(1, N - margin)
    best_idx = int(np.argmin(J_all[:search_end]))
    return best_idx, float(J_all[best_idx])


def crop_g(g_arr, cut_idx, delta_idx):
    """Remove g_arr[cut_idx : cut_idx + delta_idx] and stitch."""
    return np.concatenate([g_arr[:cut_idx], g_arr[cut_idx + delta_idx :]])


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
    Iteratively crop δ-wide windows from g to minimise ∫₀ᵀ f·g.

    Each step calls ``find_best_cut`` on the current (f, g) pair,
    then stitches g via ``crop_g`` before the next iteration.

    Returns
    -------
    sorted_real_cuts : list[float]
        Pause-start positions in original wall-clock seconds.
    J_history : list[float]
        Objective value after each accepted cut.
    baseline_J : float
        Original ∫ f·g with no pauses.
    eff_delta : float
        Effective δ after grid-snapping.
    """
    # ── Discretise f and g ─────────────────────────────────────
    x_grid = np.linspace(0, T, n_interp)
    h = x_grid[1] - x_grid[0]
    delta_idx = max(1, int(round(delta / h)))
    eff_delta = delta_idx * h
    N = n_interp

    f_arr = np.asarray(f_func(x_grid), dtype=np.float64)

    # g extends well beyond T so values can slide in as cuts accumulate
    g_len = N + (max_steps + 1) * delta_idx
    x_g = np.arange(g_len) * h
    g_arr = np.asarray(g_func(x_g), dtype=np.float64).copy()

    baseline_J = float(np.trapz(f_arr * g_arr[:N], dx=h))

    sorted_real_cuts = []
    J_history = []
    no_improve_count = 0
    best_J = baseline_J
    best_real_cuts = []

    if verbose:
        print(f"Baseline CO₂ cost (∫ f·g): {baseline_J:.4f}")
        print(
            f"δ = {delta} s  (grid: {delta_idx} cells × {h:.4f} s = {eff_delta:.4f} s)"
        )
        print(f"f grid: {N} pts on [0, {T:.0f}] s, h = {h:.4f} s")
        print(f"g grid: {g_len} pts on [0, {x_g[-1]:.0f}] s")
        print(f"max steps = {max_steps}, patience = {patience}")
        print("─" * 65)

    for step_i in range(max_steps):
        if len(g_arr) < N + delta_idx:
            if verbose:
                print(f"  Step {step_i + 1}: g exhausted, stopping.")
            break

        # ── Find the best single cut on current g ─────────────
        # During patience: exclude boundary (last delta_idx indices)
        # to avoid trivial zero-effect cuts and force interior exploration.
        m = delta_idx if no_improve_count > 0 else 0
        cut_idx, J_val = find_best_cut(f_arr, g_arr, h, delta_idx, margin=m)

        # Map compressed-domain index → original g time
        a_s = cut_idx * h
        real_a = compressed_to_real(a_s, sorted_real_cuts, eff_delta)
        sorted_real_cuts.append(real_a)
        sorted_real_cuts.sort()

        J_history.append(J_val)

        # Track best schedule seen
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
            tag = ""
            if no_improve_count > 0:
                tag = (f" [no improve {no_improve_count}/{patience}, "
                       f"current J={J_val:.4f}]")
            print(f"  Step {step_i+1}: a_real={real_a:.1f} s, "
                  f"best_J={best_J:.4f}, saved {baseline_J - best_J:.4f} "
                  f"({(baseline_J - best_J)/baseline_J*100:.2f}%){tag}")

        # ── Stitch g: always apply the cut so subsequent iterations
        #    see a different landscape (needed to escape local optima) ──
        g_arr = crop_g(g_arr, cut_idx, delta_idx)

        if no_improve_count >= patience:
            if verbose:
                print(
                    f"  Early stopping: patience exhausted. "
                    f"Reverting to best ({len(best_real_cuts)} cuts, "
                    f"J={best_J:.4f})."
                )
            break

    # Return the best tracked schedule
    sorted_real_cuts = best_real_cuts if best_real_cuts else sorted_real_cuts
    final_J = best_J

    if verbose:
        print("─" * 65)
        print(f"Best schedule: {len(sorted_real_cuts)} cuts")
        total_pause = len(sorted_real_cuts) * eff_delta
        print(
            f"Total pause time: {total_pause:.1f} s "
            f"({total_pause / T * 100:.1f}% of job)"
        )
        print(
            f"Best CO₂ cost: {final_J:.4f}  "
            f"(saved {baseline_J - final_J:.4f}, "
            f"{(baseline_J - final_J) / baseline_J * 100:.2f}%)"
        )

    return sorted_real_cuts, J_history, baseline_J, eff_delta


# ────────────────────────────────────────────────────────────────
# 5. Plotting
# ────────────────────────────────────────────────────────────────


def _merge_pauses(sorted_cuts, delta):
    """Merge overlapping/adjacent [c, c+δ) intervals into consolidated pauses."""
    if not sorted_cuts:
        return []
    merged = []
    for c in sorted(sorted_cuts):
        s, e = c, c + delta
        if merged and s <= merged[-1][1] + 1e-10:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append([s, e])
    return [(s, e) for s, e in merged]


def plot_results(f_func, g_func, T, delta, sorted_real_cuts, J_history, baseline_J):
    """
    Produce two figures.

    The pauses are shown as empty gaps in the f and f·g subplots on a
    wall-clock time axis.  g is plotted continuously (it never pauses).
    """
    cuts = sorted(sorted_real_cuts)
    pauses = _merge_pauses(cuts, delta)

    total_pause = sum(e - s for s, e in pauses)
    T_wall = T + total_pause

    # ── Active (computation-running) intervals on the wall-clock axis ──
    active = []
    prev_end = 0.0
    for ps, pe in pauses:
        if ps > prev_end + 1e-10:
            active.append((prev_end, ps))
        prev_end = pe
    if prev_end < T_wall - 1e-10:
        active.append((prev_end, T_wall))

    # ── Build f and f·g arrays with NaN gaps at pauses ─────────
    N_PLOT = 8_000
    x_parts, f_parts, fg_parts = [], [], []
    job_time = 0.0  # elapsed computation time

    for ws, we in active:
        duration = we - ws
        n_seg = max(10, int(N_PLOT * duration / T_wall))
        tw = np.linspace(ws, we, n_seg)
        tj = job_time + (tw - ws)  # map wall-clock → job time

        fv = np.asarray(f_func(tj), dtype=float)
        gv = np.asarray(g_func(tw), dtype=float)

        x_parts.append(tw)
        f_parts.append(fv)
        fg_parts.append(fv * gv)

        # NaN separator → visual break between segments
        x_parts.append([np.nan])
        f_parts.append([np.nan])
        fg_parts.append([np.nan])

        job_time += duration

    x_wall = np.concatenate(x_parts)
    f_wall = np.concatenate(f_parts)
    fg_wall = np.concatenate(fg_parts)

    # ── g is continuous on the full wall-clock axis ────────────
    x_g_cont = np.linspace(0, T_wall, N_PLOT)
    g_cont = np.asarray(g_func(x_g_cont), dtype=float)

    # ── Figure 1: three stacked subplots ───────────────────────
    fig, axes = plt.subplots(3, 1, figsize=(16, 13), sharex=True)

    plot_specs = [
        (
            x_wall,
            f_wall,
            "Power  f(t)  [kW]",
            "Power profile f(t) — pauses shown as gaps",
            "tab:blue",
            "line",
        ),
        (
            x_g_cont,
            g_cont,
            r"Carbon intensity  g(t)  [gCO$_2$eq/kWh]",
            "Carbon intensity g(t) (continuous wall-clock)",
            "tab:orange",
            "bar",
        ),
        (
            x_wall,
            fg_wall,
            r"f(t)$\cdot$g(t)  [gCO$_2$eq·s/kWh·s → gCO$_2$eq]",
            r"Product f(t)$\cdot$g(t) — pauses shown as gaps",
            "tab:green",
            "line",
        ),
    ]

    for ax_i, (xd, yd, ylabel, title, color, style) in enumerate(plot_specs):
        ax = axes[ax_i]
        if style == "bar":
            bw = xd[1] - xd[0]
            ax.bar(xd, yd, width=bw, color=color, alpha=0.7, linewidth=0)
        else:
            ax.plot(xd, yd, color=color, linewidth=1.2)
        if ax_i == 2:
            ax.fill_between(xd, yd, alpha=0.08, color=color)

        # Shade pause intervals
        for pi, (ps, pe) in enumerate(pauses):
            kw = dict(alpha=0.20, color="grey")
            if pi == 0 and ax_i == 0:
                kw["label"] = (
                    f"Pauses ({len(pauses)} merged from "
                    f"{len(cuts)} cuts, "
                    f"total {total_pause:.0f} s)"
                )
            ax.axvspan(ps, pe, **kw)

        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=12)
        if ax_i == 0:
            ax.legend(fontsize=9, loc="upper right")
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Wall-clock time  t  [s]", fontsize=11)
    fig.suptitle(
        f"CO₂-aware pause schedule: {len(cuts)} cuts (δ ≈ {delta:.0f} s)  —  "
        f"wall-clock {T_wall:.0f} s  (job {T:.0f} s + pauses {total_pause:.0f} s)",
        fontsize=13,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    # ── Figure 2: CO₂ cost vs. step number ────────────────────
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    step_range = np.arange(1, len(J_history) + 1)
    ax2.plot(
        step_range, J_history, "o-", markersize=2, color="tab:red", label="CO₂ cost"
    )
    ax2.axhline(
        baseline_J,
        color="grey",
        linestyle=":",
        alpha=0.7,
        label=f"No pauses: {baseline_J:.1f}",
    )
    ax2.set_xlabel("Optimisation step (cuts applied sequentially)", fontsize=11)
    ax2.set_ylabel(r"Total CO$_2$ cost  [gCO$_2$eq·s/kWh]", fontsize=11)
    ax2.set_title("CO₂ cost vs. optimisation step", fontsize=13)
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
    sorted_real_cuts, J_history, baseline_J, eff_delta = run_optimisation(
        f_func, g_func, T, delta=DELTA, max_steps=MAX_STEPS, patience=PATIENCE
    )

    # ── Plot ─────────────────────────────────────────────────
    print("\n[4] Plotting results …")
    plot_results(f_func, g_func, T, eff_delta, sorted_real_cuts, J_history, baseline_J)


if __name__ == "__main__":
    main()
