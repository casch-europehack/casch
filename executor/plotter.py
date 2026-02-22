import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path

matplotlib.use("Agg")

def plot_execution(samples: list, out_stem: str) -> None:
    """Save recorded samples as a PNG plot."""
    assets_dir = Path(__file__).resolve().parent.parent / "profiler" / "assets"
    assets_dir.mkdir(exist_ok=True)

    times = [s.time_s for s in samples]
    throttles = [s.throttle for s in samples]
    gpu_utils = [s.gpu_util_pct for s in samples]
    powers = [s.power_w for s in samples]

    has_gpu = any(v is not None for v in gpu_utils)
    n_panels = 3 if has_gpu else 1
    fig, axes = plt.subplots(n_panels, 1, figsize=(12, 3 * n_panels), sharex=True)

    if n_panels == 1:
        axes = [axes]

    # Panel 1 – throttle
    axes[0].step(times, throttles, where="post", color="steelblue", lw=1.8)
    axes[0].set_ylim(-0.05, 1.05)
    axes[0].set_ylabel("Throttle")
    axes[0].set_title("Policy throttle")
    axes[0].yaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.0%}")
    )
    axes[0].grid(True, alpha=0.3)

    if has_gpu:
        # Panel 2 – GPU utilisation
        axes[1].plot(times, gpu_utils, color="seagreen", lw=1.2)
        axes[1].set_ylim(-2, 105)
        axes[1].set_ylabel("GPU util (%)")
        axes[1].set_title("GPU utilisation")
        axes[1].grid(True, alpha=0.3)

        # Panel 3 – power draw
        axes[2].plot(times, powers, color="darkorange", lw=1.2)
        axes[2].set_ylabel("Power (W)")
        axes[2].set_title("GPU power draw")
        axes[2].grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time (s)")
    plt.tight_layout()

    plot_path = assets_dir / f"{out_stem}_execution.png"
    plt.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"[Monitor] Plot saved  → {plot_path}")
