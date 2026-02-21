import sys
import inspect
import importlib.util
import json
from itertools import islice
from pathlib import Path

import torch
import numpy as np
from zeus.monitor import ZeusMonitor
from models import TrainingJob, JobConfig

WARMUP_STEPS = 5


def load_job(script_path: str) -> TrainingJob:
    spec = importlib.util.spec_from_file_location("user_job", script_path)
    mod = importlib.util.load_module_from_spec(spec)
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


def profile(script_path: str):
    job = load_job(script_path)
    config: JobConfig = job.configure()

    job.setup()
    loader = job.get_dataloader()
    steps_per_epoch = len(loader)

    monitor = ZeusMonitor(gpu_indices=config.gpu_indices)

    print(f"Profiling {config.profile_steps + WARMUP_STEPS} steps (first {WARMUP_STEPS} are warmup)...")

    step_energies = []
    for i, batch in enumerate(islice(loader, config.profile_steps + WARMUP_STEPS)):
        monitor.begin_window("step")
        job.train_one_step(batch)
        result = monitor.end_window("step")

        if i < WARMUP_STEPS:
            continue  # discard warmup

        total_energy = sum(result.gpu_energy.values())
        step_energies.append(total_energy)

    job.teardown()
    return np.array(step_energies), steps_per_epoch, config


def extrapolate_and_plot(step_energies: np.ndarray, steps_per_epoch: int, config: JobConfig, out_stem: str):
    mean = step_energies.mean()
    std = step_energies.std()
    total_steps = steps_per_epoch * config.total_epochs

    xs = np.arange(1, total_steps + 1)
    cumulative      = mean * xs / 3600        # Wh
    upper           = (mean + std) * xs / 3600
    lower           = (mean - std) * xs / 3600
    epoch_marks     = [steps_per_epoch * e for e in range(1, config.total_epochs + 1)]

    # Save JSON
    output = {
        "profiled_steps": len(step_energies),
        "steps_per_epoch": steps_per_epoch,
        "total_epochs": config.total_epochs,
        "total_steps": total_steps,
        "mean_energy_per_step_J": round(float(mean), 4),
        "std_energy_per_step_J": round(float(std), 4),
        "estimated_total_energy_Wh": round(float(cumulative[-1]), 4),
        "step_energy_J": [round(e, 4) for e in step_energies.tolist()],
    }
    json_path = f"{out_stem}_profile.json"
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)

    # Plot
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    # Left: measured per-step energy during profiling
    axes[0].plot(step_energies, color="steelblue")
    axes[0].axhline(mean, color="red", linestyle="--", label=f"mean = {mean:.2f} J")
    axes[0].fill_between(range(len(step_energies)),
                         mean - std, mean + std, alpha=0.2, color="red", label="±1σ")
    axes[0].set_xlabel("Profiled step")
    axes[0].set_ylabel("Energy (J)")
    axes[0].set_title("Measured per-step energy")
    axes[0].legend()

    # Right: extrapolated cumulative energy over full job
    axes[1].plot(xs, cumulative, color="steelblue", label="Estimated cumulative (Wh)")
    axes[1].fill_between(xs, lower, upper, alpha=0.2, color="steelblue")
    for xb in epoch_marks:
        axes[1].axvline(xb, color="gray", linestyle="--", alpha=0.4)
    axes[1].set_xlabel("Training step")
    axes[1].set_ylabel("Cumulative energy (Wh)")
    axes[1].set_title(f"Projected energy — {config.total_epochs} epochs (dashed = epoch boundary)")
    axes[1].legend()

    plt.tight_layout()
    plot_path = f"{out_stem}_energy.png"
    plt.savefig(plot_path, dpi=150)
    print(f"Plot saved to {plot_path}")
    print(f"Results saved to {json_path}")
    print(f"\nEstimated total energy: {cumulative[-1]:.2f} Wh  (±{(upper[-1]-lower[-1])/2:.2f} Wh)")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python monitor.py <path_to_user_job.py>")
        sys.exit(1)

    script_path = sys.argv[1]
    out_stem = Path(script_path).stem

    step_energies, steps_per_epoch, config = profile(script_path)
    extrapolate_and_plot(step_energies, steps_per_epoch, config, out_stem)
