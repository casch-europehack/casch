import json
import os
from pathlib import Path
from typing import Any, Dict, List

from optimizer.co2_optimizer import (
    DELTA,
    MAX_STEPS,
    PATIENCE,
    create_co2_interpolator,
    create_power_function_from_data,
    fetch_and_cache_carbon_intensity,
    run_optimisation,
)
from services.storage import load_from_db

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CI_CACHE_PATH = DATA_DIR / "ci_cache.json"


def cuts_to_policy(
    sorted_real_cuts: List[float], delta: float, T: float
) -> List[Dict[str, Any]]:
    """
    Convert optimiser pause-cuts into a throttle policy.

    Each cut inserts a ``delta``-wide pause (throttle 0.0) into the
    timeline.  The segments between pauses run at full speed (1.0).

    Parameters
    ----------
    sorted_real_cuts : list[float]
        Pause start positions in *original* (g-space) seconds, sorted.
    delta : float
        Width of each pause window in seconds.
    T : float
        Total job duration in original seconds (before pauses).

    Returns
    -------
    list[dict]
        Policy segments ``[{start, end, throttle}, ...]`` in wall-clock
        seconds.
    """
    policy: List[Dict[str, Any]] = []
    wall_clock = 0.0
    prev_original = 0.0

    for cut in sorted_real_cuts:
        # Compute segment: prev_original → cut
        compute_duration = cut - prev_original
        if compute_duration > 1e-10:
            policy.append(
                {
                    "start": round(wall_clock, 2),
                    "end": round(wall_clock + compute_duration, 2),
                    "throttle": 1.0,
                }
            )
            wall_clock += compute_duration

        # Pause segment
        policy.append(
            {
                "start": round(wall_clock, 2),
                "end": round(wall_clock + delta, 2),
                "throttle": 0.0,
            }
        )
        wall_clock += delta
        prev_original = cut

    # Final compute segment after the last pause
    remaining = T - prev_original
    if remaining > 1e-10:
        policy.append(
            {
                "start": round(wall_clock, 2),
                "end": round(wall_clock + remaining, 2),
                "throttle": 1.0,
            }
        )

    return policy


class CO2Calculator:
    """Run the real CO₂-aware optimiser and return a throttle policy."""

    def calculate(
        self,
        data: Dict[str, Any],
        location: str,
        delta: float = DELTA,
        max_steps: int = MAX_STEPS,
        patience: int = PATIENCE,
    ) -> Dict[str, Any]:
        # 1. Build the power function f(t) from profiler data
        f_func, step_boundaries, _step_powers = create_power_function_from_data(data)
        run_duration = float(step_boundaries[-1])
        T = run_duration

        # 2. Build the carbon-intensity function g(t)
        #    Use a per-zone cache so repeated calls are fast.
        zone_cache = DATA_DIR / f"ci_cache_{location}.json"
        ci_times, ci_values = fetch_and_cache_carbon_intensity(
            zone_cache, zone=location
        )
        g_func = create_co2_interpolator(ci_times, ci_values)

        # 3. Run the optimisation
        sorted_real_cuts, J_history, baseline_J, eff_delta = run_optimisation(
            f_func,
            g_func,
            T,
            g_end_time=float(ci_times[-1]),
            delta=delta,
            max_steps=max_steps,
            patience=patience,
            verbose=False,
        )

        # 4. Derive final CO₂ cost
        final_J = J_history[-1] if J_history else baseline_J
        savings_abs = baseline_J - final_J
        savings_pct = (savings_abs / baseline_J * 100) if baseline_J else 0.0

        # 5. Convert cuts → throttle policy (use grid-snapped eff_delta)
        policy = cuts_to_policy(sorted_real_cuts, eff_delta, T)

        return {
            "location": location,
            "job_duration_s": round(T, 2),
            "delta_s": round(eff_delta, 4),
            "num_pauses": len(sorted_real_cuts),
            "baseline_co2": round(baseline_J, 4),
            "optimised_co2": round(final_J, 4),
            "savings_co2": round(savings_abs, 4),
            "savings_pct": round(savings_pct, 2),
            "pause_positions_s": [round(c, 2) for c in sorted_real_cuts],
            "ci_times_s": [round(float(t), 2) for t in ci_times],
            "ci_values": [round(float(v), 2) for v in ci_values],
            "policy": policy,
        }


class CO2ProxyService:
    def __init__(self):
        self.calculator = CO2Calculator()

    def get_co2_emissions(self, file_hash: str, location: str) -> Dict[str, Any]:
        data = load_from_db(file_hash)
        if not data:
            raise ValueError("Data not found for the given file hash")

        result = self.calculator.calculate(data, location)

        # Persist the policy so the /schedule endpoint can use it
        policies_path = f"{file_hash}_policies.json"
        policies: Dict[str, Any] = {}
        if os.path.exists(policies_path):
            with open(policies_path, "r") as f:
                policies = json.load(f)

        policies["baseline"] = {
            "policy": [{"start": 0.0, "end": result["job_duration_s"], "throttle": 1.0}]
        }
        policies["co2_optimised"] = {"policy": result["policy"]}

        with open(policies_path, "w") as f:
            json.dump(policies, f, indent=2)

        return result

    def get_aggregate(self, file_hash: str, location: str) -> Dict[str, Any]:
        """
        Run the optimiser and return the duration-vs-CO2 trade-off curve.

        Each optimisation step inserts one additional pause of width
        ``eff_delta``, increasing wall-clock duration while (generally)
        reducing CO2.  The returned arrays let the frontend plot
        *duration* on the X-axis and *CO2 emissions* on the Y-axis.

        A schedulable policy is generated and stored for **every**
        optimisation step so that any point on the curve can be
        executed via the ``/schedule`` endpoint.

        Returns
        -------
        dict with keys:
            file_hash, location, job_duration_s, delta_s,
            num_optimal_pauses, baseline_co2, optimised_co2,
            durations, co2_emissions, policy_ids
        """
        data = load_from_db(file_hash)
        if not data:
            raise ValueError("Data not found for the given file hash")

        # 1. Build f(t) and g(t)
        f_func, step_boundaries, _ = create_power_function_from_data(data)
        T = float(step_boundaries[-1])

        zone_cache = DATA_DIR / f"ci_cache_{location}.json"
        ci_times, ci_values = fetch_and_cache_carbon_intensity(
            zone_cache, zone=location
        )
        g_func = create_co2_interpolator(ci_times, ci_values)

        # 2. Run the optimisation, recording intermediate cut snapshots
        (
            sorted_real_cuts,
            J_history,
            baseline_J,
            eff_delta,
            cuts_history,
        ) = run_optimisation(
            f_func,
            g_func,
            T,
            g_end_time=float(ci_times[-1]),
            delta=DELTA,
            max_steps=MAX_STEPS,
            patience=PATIENCE,
            verbose=False,
            record_cuts_history=True,
        )

        # 3. Build the trade-off curve from J_history
        #    J_history[k] is the CO2 cost after step k+1.
        #    Duration at step k+1 = T + (k+1) * eff_delta.
        #    We track a running minimum so the curve reflects the
        #    best achievable CO2 at each duration level.
        durations: List[float] = [round(T, 2)]
        co2_emissions: List[float] = [round(baseline_J, 4)]
        policy_ids: List[str] = ["baseline"]

        running_min = baseline_J
        for k, j_val in enumerate(J_history):
            running_min = min(running_min, j_val)
            durations.append(round(float(T + (k + 1) * eff_delta), 2))
            co2_emissions.append(round(float(running_min), 4))
            policy_ids.append(f"step_{k + 1}")

        # 4. Generate & store a policy for every optimisation step
        n_optimal = len(sorted_real_cuts)
        optimal_co2 = co2_emissions[-1] if co2_emissions else round(baseline_J, 4)

        policies_path = f"{file_hash}_policies.json"
        policies: Dict[str, Any] = {}
        if os.path.exists(policies_path):
            with open(policies_path, "r") as f:
                policies = json.load(f)

        # Baseline (no pauses)
        policies["baseline"] = {
            "policy": [{"start": 0.0, "end": round(T, 2), "throttle": 1.0}]
        }

        # One policy per optimisation step
        for k, step_cuts in enumerate(cuts_history):
            key = f"step_{k + 1}"
            policies[key] = {
                "policy": cuts_to_policy(step_cuts, eff_delta, T),
                "num_pauses": len(step_cuts),
                "duration_s": round(float(T + len(step_cuts) * eff_delta), 2),
            }

        # Alias the final optimal policy for backwards compatibility
        policies["co2_optimised"] = {
            "policy": cuts_to_policy(sorted_real_cuts, eff_delta, T),
        }

        with open(policies_path, "w") as f:
            json.dump(policies, f, indent=2)

        return {
            "file_hash": file_hash,
            "location": location,
            "job_duration_s": round(T, 2),
            "delta_s": round(eff_delta, 4),
            "num_optimal_pauses": n_optimal,
            "baseline_co2": round(baseline_J, 4),
            "optimised_co2": optimal_co2,
            "durations": durations,
            "co2_emissions": co2_emissions,
            "policy_ids": policy_ids,
        }


co2_service = CO2ProxyService()
