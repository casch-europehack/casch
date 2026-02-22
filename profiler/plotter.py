import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from utils.converters import aggregate_intervals

def _time_axis(seconds: np.ndarray):
    """Return (values, unit) choosing ms or s for readability."""
    if seconds.max() < 1.0:
        return seconds * 1000, "ms"
    return seconds, "s"

def plot(output: dict, out_stem: str):
    step_energies = np.array(output["profiled_step_energy_J"])
    step_times = np.array(output["profiled_step_time_s"])
    pred_energies = np.array(output["step_energy_J"])
    pred_times = np.array(output["step_time_s"])
    energy_mean = output["mean_energy_per_step_J"]
    energy_std = output["std_energy_per_step_J"]
    profile_epochs = output["profiled_epochs"]
    total_epochs = output["total_epochs"]
    total_steps = output["total_steps"]
    steps_per_epoch = output["steps_per_epoch"]
    time_mean = output["mean_time_per_step_s"]
    estimated_total_energy_Wh = output["estimated_total_energy_Wh"]
    estimated_total_time_s = output["estimated_total_time_s"]

    upper_energy = (energy_mean + energy_std) * np.arange(1, total_steps + 1) / 3600
    lower_energy = (energy_mean - energy_std) * np.arange(1, total_steps + 1) / 3600

    assets_dir = Path(__file__).resolve().parent.parent / "profiler" / "assets"
    assets_dir.mkdir(exist_ok=True)
    json_path = assets_dir / f"{out_stem}_profile.json"
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Results saved to {json_path}")
    print(f"\nEstimated total energy: {estimated_total_energy_Wh:.2f} Wh  (±{(upper_energy[-1]-lower_energy[-1])/2:.2f} Wh)")
    print(f"Estimated total time: {estimated_total_time_s:.1f} s")

    epoch_times_s = np.array([steps_per_epoch * e * time_mean for e in range(1, total_epochs + 1)])

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
    axes[0].legend(loc="upper right")

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
    axes[1].set_title(f"Projected per-step energy — {total_epochs} epochs")
    axes[1].legend(loc="upper right")

    plt.tight_layout()
    plot_path = assets_dir / f"{out_stem}_energy.png"
    plt.savefig(plot_path, dpi=150)
    print(f"Plot saved to {plot_path}")


def plot_power(output: dict, out_stem: str):
    step_energies = np.array(output["profiled_step_energy_J"])
    step_times = np.array(output["profiled_step_time_s"])
    pred_energies = np.array(output["step_energy_J"])
    pred_times = np.array(output["step_time_s"])
    profile_epochs = output["profiled_epochs"]
    total_epochs = output["total_epochs"]
    steps_per_epoch = output["steps_per_epoch"]
    time_mean = output["mean_time_per_step_s"]

    # Aggregate intervals for the projected power to improve plot performance and readability
    agg_pred_energies, agg_pred_times = aggregate_intervals(pred_energies, pred_times, num_blocks=200)

    # Calculate power (P = E / t)
    step_power = step_energies / step_times
    pred_power = agg_pred_energies / agg_pred_times
    
    power_mean = step_power.mean()
    power_std = step_power.std()

    epoch_times_s = np.array([steps_per_epoch * e * time_mean for e in range(1, total_epochs + 1)])

    assets_dir = Path(__file__).resolve().parent.parent / "profiler" / "assets"
    assets_dir.mkdir(exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    # Left: measured per-step power during profiling
    profiled_times_cum = np.cumsum(step_times)
    ptime, punit = _time_axis(profiled_times_cum)
    
    # Use step plot for power
    axes[0].step(ptime, step_power, where='post', color="darkorange")
    axes[0].axhline(power_mean, color="red", linestyle="--", label=f"mean = {power_mean:.2f} W")
    axes[0].fill_between(ptime,
                         power_mean - power_std, power_mean + power_std,
                         alpha=0.2, color="red", label="±1σ", step='post')
    axes[0].set_xlabel(f"Time ({punit})")
    axes[0].set_ylabel("Power (W)")
    axes[0].set_title(f"Measured per-step power — {profile_epochs} epoch(s) profiled")
    axes[0].legend(loc="upper right")

    # Right: tiled per-step power extrapolated over full training
    ext_times_cum = np.cumsum(agg_pred_times)
    ext_time, ext_unit = _time_axis(ext_times_cum)
    epoch_marks, _ = _time_axis(epoch_times_s)
    
    axes[1].step(ext_time, pred_power, where='post', color="darkorange", alpha=0.6)
    axes[1].axhline(power_mean, color="red", linestyle="--", label=f"mean = {power_mean:.2f} W")
    axes[1].fill_between(ext_time,
                         power_mean - power_std, power_mean + power_std,
                         alpha=0.2, color="red", label="±1σ", step='post')
    for xb in epoch_marks:
        axes[1].axvline(xb, color="gray", linestyle="--", alpha=0.4)
    axes[1].set_xlabel(f"Time ({ext_unit})")
    axes[1].set_ylabel("Power (W)")
    axes[1].set_title(f"Projected per-step power — {total_epochs} epochs")
    axes[1].legend(loc="upper right")

    plt.tight_layout()
    plot_path = assets_dir / f"{out_stem}_power.png"
    plt.savefig(plot_path, dpi=150)
    print(f"Power plot saved to {plot_path}")
