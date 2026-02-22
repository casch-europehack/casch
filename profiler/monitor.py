import sys
import argparse
import importlib.util
import inspect
import time
from pathlib import Path

_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import numpy as np
from profiler.models import TrainingJob, JobConfig
from profiler.plotter import plot, plot_power

# Number of initial steps to discard as warmup (e.g. for JIT compilation, caching, etc.)
WARMUP_STEPS = 5

# Fallback TDP in watts for time-based energy estimates when no GPU is available (e.g. CPU-only). Adjust as needed.
DEFAULT_TDP_W = 250

class _FallbackResult:
    """Time-based energy estimate when Zeus is unavailable (no NVIDIA/AMD GPU)."""
    def __init__(self, elapsed_s: float, tdp_w: float):
        self.gpu_energy = {0: elapsed_s * tdp_w}


class _FallbackMonitor:
    def __init__(self, tdp_w: float = DEFAULT_TDP_W):
        self._tdp_w = tdp_w
        self._start = None

    def begin_window(self, _name: str):
        self._start = time.perf_counter()

    def end_window(self, _name: str):
        elapsed = time.perf_counter() - self._start
        return _FallbackResult(elapsed, self._tdp_w)

def _load_job(script_path: str) -> TrainingJob:
    """
    Dynamically load the user script and find the TrainingJob subclass.
    
    Args:
        script_path: Path to the user job script.
    Returns:
        An instance of the TrainingJob subclass defined in the user script.
    Raises:
        ValueError: If no TrainingJob subclass is found or if multiple are found.
    """
    spec = importlib.util.spec_from_file_location("user_job", script_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    subclasses = [
        cls for _, cls in inspect.getmembers(mod, inspect.isclass)
        if issubclass(cls, TrainingJob) and cls is not TrainingJob
    ]
    if not subclasses:
        raise ValueError("No TrainingJob subclass found.")
    if len(subclasses) > 1:
        raise ValueError(f"Multiple TrainingJob subclasses found: {subclasses}")

    return subclasses[0]()


def profile(script_path: str, profile_epochs: int = 1, warmup_steps: int = WARMUP_STEPS, tdp_w: float = DEFAULT_TDP_W):
    job = _load_job(script_path)
    config: JobConfig = job.configure()

    job.setup()
    loader = job.get_dataloader()
    steps_per_epoch = len(loader)
    total_profile_steps = steps_per_epoch * profile_epochs

    try:
        from zeus.monitor import ZeusMonitor
        monitor = ZeusMonitor(gpu_indices=config.gpu_indices)
    except Exception:
        print("WARNING: No supported GPU found, using time-based energy estimates.")
        monitor = _FallbackMonitor(tdp_w=tdp_w)

    print(f"Profiling {total_profile_steps + warmup_steps} steps "
          f"({profile_epochs} epoch(s), first {warmup_steps} are warmup)...")

    step_energies = []
    step_times = []
    step = 0
    for epoch in range(profile_epochs):
        for batch in loader:
            t0 = time.perf_counter()
            monitor.begin_window("step")
            job.train_one_step(batch)
            result = monitor.end_window("step")
            elapsed = time.perf_counter() - t0

            if step < warmup_steps:
                step += 1
                continue

            total_energy = sum(result.gpu_energy.values())
            step_energies.append(total_energy)
            step_times.append(elapsed)
            step += 1

    job.teardown()
    return np.array(step_energies), np.array(step_times), steps_per_epoch, config


def extrapolate(
    step_energies: np.ndarray,
    step_times: np.ndarray,
    steps_per_epoch: int,
    config: JobConfig,
    out_stem: str,
    profile_epochs: int = 1,
):
    energy_mean = step_energies.mean()
    energy_std = step_energies.std()
    time_mean = step_times.mean()
    time_std = step_times.std()
    total_steps = steps_per_epoch * config.total_epochs

    cum_time_s = time_mean * np.arange(1, total_steps + 1)
    cumulative_energy = energy_mean * np.arange(1, total_steps + 1) / 3600

    reps = int(np.ceil(total_steps / len(step_energies)))
    pred_energies = np.tile(step_energies, reps)[:total_steps]
    pred_times = np.tile(step_times, reps)[:total_steps]

    output = {
        "profiled_epochs": profile_epochs,
        "steps_per_epoch": steps_per_epoch,
        "total_epochs": config.total_epochs,
        "total_steps": total_steps,
        "mean_energy_per_step_J": round(float(energy_mean), 4),
        "std_energy_per_step_J": round(float(energy_std), 4),
        "mean_time_per_step_s": round(float(time_mean), 6),
        "std_time_per_step_s": round(float(time_std), 6),
        "estimated_total_energy_Wh": round(float(cumulative_energy[-1]), 4),
        "estimated_total_time_s": round(float(cum_time_s[-1]), 2),
        "profiled_step_energy_J": [round(e, 4) for e in step_energies.tolist()],
        "profiled_step_time_s": [round(t, 6) for t in step_times.tolist()],
        "step_energy_J": [round(e, 4) for e in pred_energies.tolist()],
        "step_time_s": [round(t, 6) for t in pred_times.tolist()],
    }

    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Profile a training job's energy usage.")
    parser.add_argument("script", help="Path to the user job script")
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs to profile (default: 1)")
    parser.add_argument("--steps", type=int, default=WARMUP_STEPS, help=f"Warmup steps to discard (default: {WARMUP_STEPS})")
    parser.add_argument("--default-tdp-w", type=float, default=DEFAULT_TDP_W, help=f"Fallback TDP in watts when no GPU is available (default: {DEFAULT_TDP_W})")
    args = parser.parse_args()

    out_stem = Path(args.script).stem

    step_energies, step_times, steps_per_epoch, config = profile(args.script, profile_epochs=args.epochs, warmup_steps=args.steps, tdp_w=args.default_tdp_w)
    output = extrapolate(step_energies, step_times, steps_per_epoch, config, out_stem, profile_epochs=args.epochs)
    
    plot(output, out_stem)
    plot_power(output, out_stem)
