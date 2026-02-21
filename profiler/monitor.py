import sys
import argparse
import importlib.util
import inspect
import json
import time
from pathlib import Path

_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import matplotlib.pyplot as plt
import numpy as np
import torch
from profiler.models import TrainingJob, JobConfig

WARMUP_STEPS = 5
DEFAULT_TDP_W = 250


def _zeus_available():
    try:
        from zeus.monitor import ZeusMonitor
        ZeusMonitor(gpu_indices=[0])
        return True
    except Exception:
        return False


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


def load_job(script_path: str) -> TrainingJob:
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
    job = load_job(script_path)
    config: JobConfig = job.configure()

    job.setup()
    loader = job.get_dataloader()
    steps_per_epoch = len(loader)
    total_profile_steps = steps_per_epoch * profile_epochs

    if _zeus_available():
        from zeus.monitor import ZeusMonitor
        monitor = ZeusMonitor(gpu_indices=config.gpu_indices)
    else:
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


def _time_axis(seconds: np.ndarray):
    """Return (values, unit) choosing ms or s for readability."""
    if seconds.max() < 1.0:
        return seconds * 1000, "ms"
    return seconds, "s"


def extrapolate_and_plot(
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
    upper_energy = (energy_mean + energy_std) * np.arange(1, total_steps + 1) / 3600
    lower_energy = (energy_mean - energy_std) * np.arange(1, total_steps + 1) / 3600
    epoch_times_s = np.array([steps_per_epoch * e * time_mean for e in range(1, config.total_epochs + 1)])

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
    assets_dir = Path(__file__).resolve().parent / "assets"
    assets_dir.mkdir(exist_ok=True)
    json_path = assets_dir / f"{out_stem}_profile.json"
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    # Left: measured per-step energy during profiling
    profiled_times_cum = np.cumsum(step_times)
    ptime, punit = _time_axis(profiled_times_cum)
    axes[0].plot(ptime, step_energies, color="steelblue")
    axes[0].axhline(energy_mean, color="red", linestyle="--", label=f"mean = {energy_mean:.2f} J")
    axes[0].fill_between(ptime,
                         energy_mean - energy_std, energy_mean + energy_std,
                         alpha=0.2, color="red", label="±1σ")
    axes[0].set_xlabel(f"Time ({punit})")
    axes[0].set_ylabel("Energy (J)")
    axes[0].set_title(f"Measured per-step energy — {profile_epochs} epoch(s) profiled")
    axes[0].legend()

    # Right: tiled per-step energy extrapolated over full training
    ext_times_cum = np.cumsum(pred_times)
    ext_time, ext_unit = _time_axis(ext_times_cum)
    epoch_marks, _ = _time_axis(epoch_times_s)
    axes[1].plot(ext_time, pred_energies, color="steelblue", alpha=0.6)
    axes[1].axhline(energy_mean, color="red", linestyle="--", label=f"mean = {energy_mean:.2f} J")
    axes[1].fill_between(ext_time,
                         energy_mean - energy_std, energy_mean + energy_std,
                         alpha=0.2, color="red", label="±1σ")
    for xb in epoch_marks:
        axes[1].axvline(xb, color="gray", linestyle="--", alpha=0.4)
    axes[1].set_xlabel(f"Time ({ext_unit})")
    axes[1].set_ylabel("Energy (J)")
    axes[1].set_title(f"Projected per-step energy — {config.total_epochs} epochs")
    axes[1].legend()

    plt.tight_layout()
    plot_path = assets_dir / f"{out_stem}_energy.png"
    plt.savefig(plot_path, dpi=150)
    print(f"Plot saved to {plot_path}")
    print(f"Results saved to {json_path}")
    print(f"\nEstimated total energy: {cumulative_energy[-1]:.2f} Wh  (±{(upper_energy[-1]-lower_energy[-1])/2:.2f} Wh)")
    print(f"Estimated total time: {cum_time_s[-1]:.1f} s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Profile a training job's energy usage.")
    parser.add_argument("script", help="Path to the user job script")
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs to profile (default: 1)")
    parser.add_argument("--steps", type=int, default=WARMUP_STEPS, help=f"Warmup steps to discard (default: {WARMUP_STEPS})")
    parser.add_argument("--default-tdp-w", type=float, default=DEFAULT_TDP_W, help=f"Fallback TDP in watts when no GPU is available (default: {DEFAULT_TDP_W})")
    args = parser.parse_args()

    out_stem = Path(args.script).stem

    step_energies, step_times, steps_per_epoch, config = profile(args.script, profile_epochs=args.epochs, warmup_steps=args.steps, tdp_w=args.default_tdp_w)
    extrapolate_and_plot(step_energies, step_times, steps_per_epoch, config, out_stem, profile_epochs=args.epochs)
