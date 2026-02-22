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

        policies["co2_optimised"] = {"policy": result["policy"]}

        with open(policies_path, "w") as f:
            json.dump(policies, f, indent=2)

        return result


co2_service = CO2ProxyService()
