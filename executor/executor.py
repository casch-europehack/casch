"""
executor.py – GPU-throttled training job executor.

Policy
------
A policy is a list of PolicySegment(start, end, throttle) where
  - start / end   are seconds measured from the moment run() is called
  - throttle       is a float in [0.0, 1.0]
      0.0  → GPU is idle; training is paused (no steps executed)
      1.0  → GPU runs at full speed (no sleep inserted)
      0.5  → GPU runs half the time (sleep equal to step time after each step)

Throttle mechanism: on/off duty-cycle
--------------------------------------
After each training step a sleep is inserted so that:

    active_time / (active_time + sleep_time) = throttle

    => sleep_time = step_time * (1 - throttle) / throttle

This is a pure software approach – the GPU runs at full speed while active
and is simply idle during the sleep.  No hardware/driver permissions needed.

How does throttle affect speed?
---------------------------------
    wall_clock_time = full_speed_time / throttle

    throttle 1.00  →  1.0x  (no change)
    throttle 0.75  →  1.33x slower
    throttle 0.50  →  2.0x  slower
    throttle 0.25  →  4.0x  slower
    throttle 0.00  →  paused entirely (no steps run)
"""

from __future__ import annotations

import argparse
import importlib.util
import inspect
import json
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch

_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import numpy as np
from profiler.models import TrainingJob, JobConfig  # noqa: E402
from executor.plotter import plot_execution

try:
    import pynvml as _pynvml

    _pynvml.nvmlInit()
    _NVML_OK = True
except Exception:
    _NVML_OK = False


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class PolicySegment:
    """
    A time window during which the GPU runs at *throttle* fraction of full speed.

    Attributes
    ----------
    start:
        Seconds after job start when this segment becomes active.
    end:
        Seconds after job start when this segment ends (exclusive).
    throttle:
        Fraction of time the GPU is actively running.
        0.0 = fully paused, 1.0 = full speed, 0.5 = on half the time.
    """

    start: float
    end: float
    throttle: float

    def __post_init__(self) -> None:
        if not (0.0 <= self.throttle <= 1.0):
            raise ValueError(f"throttle must be in [0.0, 1.0], got {self.throttle}")
        if self.start >= self.end:
            raise ValueError(
                f"start must be strictly less than end, got {self.start} >= {self.end}"
            )


# ---------------------------------------------------------------------------
# Execution monitor
# ---------------------------------------------------------------------------


@dataclass
class _Sample:
    time_s: float
    throttle: float
    gpu_util_pct: Optional[float]
    power_w: Optional[float]


class ExecutionMonitor:
    """
    Polls GPU utilisation and power draw in a background thread while a job
    runs, then saves a plot and JSON file as proof of what the GPU was doing.

    If no NVIDIA GPU / pynvml is available the monitor still records
    timestamps and throttle values so at least the policy trace is visible.

    Parameters
    ----------
    gpu_index:
        Which GPU to sample (default: 0).
    poll_interval:
        Seconds between samples (default: 0.2 s → 5 samples/s).
    """

    def __init__(self, gpu_index: int = 0, poll_interval: float = 0.2) -> None:
        self._gpu_index = gpu_index
        self._poll_interval = poll_interval
        self._samples: list[_Sample] = []
        self._throttle: float = 1.0
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._start_time: float = 0.0

        self._handle = None
        if _NVML_OK:
            try:
                self._handle = _pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
                name = _pynvml.nvmlDeviceGetName(self._handle)
                print(f"[Monitor] GPU {gpu_index}: {name}")
            except Exception as exc:
                print(
                    f"[Monitor] pynvml handle failed ({exc}); recording throttle only."
                )
                self._handle = None

    # ------------------------------------------------------------------
    # Thread interface
    # ------------------------------------------------------------------

    def start(self) -> None:
        self._start_time = time.perf_counter()
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join()

    def set_throttle(self, throttle: float) -> None:
        with self._lock:
            self._throttle = throttle

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _poll_loop(self) -> None:
        while not self._stop_event.is_set():
            elapsed = time.perf_counter() - self._start_time

            with self._lock:
                throttle = self._throttle

            gpu_util: Optional[float] = None
            power_w: Optional[float] = None

            if self._handle is not None:
                try:
                    rates = _pynvml.nvmlDeviceGetUtilizationRates(self._handle)
                    gpu_util = float(rates.gpu)
                    power_w = _pynvml.nvmlDeviceGetPowerUsage(self._handle) / 1000.0
                except Exception:
                    pass

            self._samples.append(
                _Sample(
                    time_s=round(elapsed, 3),
                    throttle=throttle,
                    gpu_util_pct=gpu_util,
                    power_w=power_w,
                )
            )

            time.sleep(self._poll_interval)

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------

    def save(self, out_stem: str, config: Optional[JobConfig] = None, steps_per_epoch: Optional[int] = None) -> None:
        """Save recorded samples as JSON and a PNG plot."""
        assets_dir = Path(__file__).resolve().parent.parent / "profiler" / "assets"
        assets_dir.mkdir(exist_ok=True)

        # ── JSON ────────────────────────────────────────────────────────
        json_path = assets_dir / f"{out_stem}_execution.json"
        
        # Transform execution data to match extrapolate output format
        step_energies = []
        step_times = []
        throttles = []
        for i in range(1, len(self._samples)):
            dt = self._samples[i].time_s - self._samples[i-1].time_s
            if dt <= 0:
                continue
            power = self._samples[i].power_w
            if power is None:
                power = 0.0
            step_times.append(dt)
            step_energies.append(power * dt)
            throttles.append(self._samples[i].throttle)
            
        energy_mean = np.mean(step_energies) if step_energies else 0.0
        energy_std = np.std(step_energies) if step_energies else 0.0
        time_mean = np.mean(step_times) if step_times else 0.0
        time_std = np.std(step_times) if step_times else 0.0
        total_steps = len(step_energies)
        
        total_epochs = config.total_epochs if config else 1
        spe = steps_per_epoch if steps_per_epoch else total_steps
            
        output = {
            "profiled_epochs": 1,
            "steps_per_epoch": spe,
            "total_epochs": total_epochs,
            "total_steps": total_steps,
            "mean_energy_per_step_J": round(float(energy_mean), 4),
            "std_energy_per_step_J": round(float(energy_std), 4),
            "mean_time_per_step_s": round(float(time_mean), 6),
            "std_time_per_step_s": round(float(time_std), 6),
            "estimated_total_energy_Wh": round(sum(step_energies) / 3600, 4),
            "estimated_total_time_s": round(sum(step_times), 2),
            "profiled_step_energy_J": [round(e, 4) for e in step_energies],
            "profiled_step_time_s": [round(t, 6) for t in step_times],
            "step_energy_J": [round(e, 4) for e in step_energies],
            "step_time_s": [round(t, 6) for t in step_times],
            "throttles": throttles
        }
        
        with open(json_path, "w") as f:
            json.dump(output, f, indent=2)
        print(f"[Monitor] Data saved → {json_path}")

        # ── Plot ────────────────────────────────────────────────────────


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_job(script_path: str) -> TrainingJob:
    spec = importlib.util.spec_from_file_location("user_job", script_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    subclasses = [
        cls
        for _, cls in inspect.getmembers(mod, inspect.isclass)
        if issubclass(cls, TrainingJob) and cls is not TrainingJob
    ]
    if not subclasses:
        raise ValueError("No TrainingJob subclass found in the script.")
    if len(subclasses) > 1:
        raise ValueError(f"Multiple TrainingJob subclasses found: {subclasses}")
    return subclasses[0]()


# ---------------------------------------------------------------------------
# Executor
# ---------------------------------------------------------------------------


class Executor:
    """
    Runs a :class:`~profiler.models.TrainingJob` while applying a throttle policy.

    Throttling is implemented as a duty-cycle: the GPU runs each step at full
    speed, then sleeps for long enough that the active fraction equals *throttle*.
    A throttle of 0.0 pauses training entirely (no steps are executed) until
    the policy moves to a non-zero segment.

    Parameters
    ----------
    policy:
        A list of :class:`PolicySegment` objects.  Segments must not overlap.
        - Before the first segment: full speed (throttle = 1.0).
        - After the last segment ends: the last segment's throttle is held for
          the remainder of the job.
    gpu_indices:
        Kept for API compatibility.  Defaults to ``[0]``.
    monitor:
        An optional :class:`ExecutionMonitor` to record GPU metrics during the
        run.  If provided, the executor keeps the monitor's throttle value in
        sync so the recorded trace matches the policy exactly.

    Examples
    --------
    >>> policy = [
    ...     PolicySegment(start=0,   end=60,  throttle=1.0),
    ...     PolicySegment(start=60,  end=120, throttle=0.5),
    ...     PolicySegment(start=120, end=180, throttle=0.0),
    ...     PolicySegment(start=180, end=300, throttle=0.75),
    ... ]
    >>> monitor = ExecutionMonitor(gpu_index=0)
    >>> executor = Executor(policy=policy, monitor=monitor)
    >>> executor.run(my_training_job)
    >>> monitor.save("my_job")
    """

    def __init__(
        self,
        policy: list[PolicySegment],
        gpu_indices: Optional[list[int]] = None,
        monitor: Optional[ExecutionMonitor] = None,
    ) -> None:
        self.policy = sorted(policy, key=lambda s: s.start)
        self.gpu_indices = gpu_indices if gpu_indices is not None else [0]
        self.monitor = monitor

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _throttle_at(self, elapsed: float) -> float:
        """Return the throttle value that applies at *elapsed* seconds.

        - Before the first segment: full speed (1.0).
        - Inside a segment: that segment's throttle.
        - After the last segment ends: the last segment's throttle is held,
          so the job finishes at whatever rate was in effect at the end of
          the policy rather than suddenly jumping back to full speed.
        """
        last: Optional[PolicySegment] = None
        for seg in self.policy:
            if seg.start <= elapsed < seg.end:
                return seg.throttle
            if elapsed >= seg.end:
                last = seg
        if last is not None:
            return last.throttle
        return 1.0  # before the first segment → full speed

    @staticmethod
    def _duty_cycle_sleep(step_time: float, throttle: float) -> None:
        """Sleep so that active / (active + sleep) == throttle.

        Derivation:
            sleep = step_time * (1 - throttle) / throttle
        """
        if throttle >= 1.0:
            return
        time.sleep(step_time * (1.0 - throttle) / throttle)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, job: TrainingJob) -> tuple[JobConfig, int]:
        """
        Execute *job* under the configured throttle policy.

        Parameters
        ----------
        job:
            A :class:`~profiler.models.TrainingJob` instance to execute.
        """
        config = job.configure()
        job.setup()
        loader = job.get_dataloader()

        if self.monitor is not None:
            self.monitor.start()

        job_start = time.perf_counter()
        current_throttle: float = -1.0  # sentinel to trigger first update

        try:
            for epoch in range(config.total_epochs):
                print(f"[Executor] Epoch {epoch + 1}/{config.total_epochs}")

                for batch in loader:
                    elapsed = time.perf_counter() - job_start
                    throttle = self._throttle_at(elapsed)

                    # ── 1. Notify monitor and log on throttle change ─────────
                    if throttle != current_throttle:
                        current_throttle = throttle
                        if self.monitor is not None:
                            self.monitor.set_throttle(throttle)
                        print(
                            f"[Executor] t={elapsed:.1f}s – throttle → {throttle:.2f}"
                        )

                    # ── 2. throttle = 0 → pause until policy changes ─────────
                    while throttle == 0.0:
                        time.sleep(0.05)
                        elapsed = time.perf_counter() - job_start
                        throttle = self._throttle_at(elapsed)
                        if throttle != current_throttle:
                            current_throttle = throttle
                            if self.monitor is not None:
                                self.monitor.set_throttle(throttle)
                            print(
                                f"[Executor] t={elapsed:.1f}s – throttle → {throttle:.2f}"
                            )

                    # ── 3. Execute one training step ─────────────────────────
                    if throttle >= 1.0:
                        # Full speed: skip synchronize and sleep entirely so
                        # PyTorch's async CPU/GPU overlap is completely preserved.
                        job.train_one_step(batch)
                    else:
                        t0 = time.perf_counter()
                        job.train_one_step(batch)
                        # Block until the GPU has actually finished all queued kernels.
                        # Without this, train_one_step() returns almost instantly
                        # (PyTorch GPU ops are async), so step_time ≈ 0 and the
                        # duty-cycle sleep becomes ≈ 0 too — leaving the GPU at 100%.
                        if torch.cuda.is_available():
                            torch.cuda.synchronize()
                        step_time = time.perf_counter() - t0

                        # ── 4. Sleep to achieve the duty-cycle fraction ──────
                        self._duty_cycle_sleep(step_time, throttle)

        finally:
            if self.monitor is not None:
                self.monitor.stop()
            job.teardown()
            
        return config, len(loader)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run a training job with a GPU throttle policy.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Policy format (JSON array of segments):
  '[{"start": 0, "end": 60, "throttle": 1.0},
    {"start": 60, "end": 120, "throttle": 0.5},
    {"start": 120, "end": 180, "throttle": 0.0}]'

  start / end  – seconds from job start
  throttle     – 0.0 (paused) to 1.0 (full speed)

Examples:
  python -m executor.executor jobs/test_job.py --policy '[{"start":0,"end":300,"throttle":1.0}]'
  python -m executor.executor jobs/test_job.py --policy-file policy.json --out my_run
        """,
    )
    parser.add_argument(
        "script",
        help="Path to the user job script containing a TrainingJob subclass.",
    )
    parser.add_argument(
        "--policy",
        type=str,
        default=None,
        help="JSON string describing the throttle policy (array of {start, end, throttle}).",
    )
    parser.add_argument(
        "--policy-file",
        type=str,
        default=None,
        help="Path to a JSON file describing the throttle policy.",
    )
    parser.add_argument(
        "--gpus",
        type=int,
        nargs="+",
        default=[0],
        metavar="IDX",
        help="GPU indices (default: 0).",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help=(
            "Output file stem for the monitor plot and JSON "
            "(default: the job script name). "
            "Files are saved to profiler/assets/."
        ),
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=0.2,
        metavar="SEC",
        help="Seconds between GPU metric samples (default: 0.2).",
    )

    args = parser.parse_args()

    if args.policy and args.policy_file:
        parser.error("Provide either --policy or --policy-file, not both.")
    if not args.policy and not args.policy_file:
        parser.error("One of --policy or --policy-file is required.")

    if args.policy_file:
        with open(args.policy_file) as f:
            raw_policy = json.load(f)
    else:
        raw_policy = json.loads(args.policy)

    policy = [
        PolicySegment(
            start=float(seg["start"]),
            end=float(seg["end"]),
            throttle=float(seg["throttle"]),
        )
        for seg in raw_policy
    ]

    out_stem = args.out or Path(args.script).stem

    monitor = ExecutionMonitor(gpu_index=args.gpus[0], poll_interval=args.poll_interval)
    job = _load_job(args.script)
    executor = Executor(policy=policy, gpu_indices=args.gpus, monitor=monitor)
    config, steps_per_epoch = executor.run(job)
    monitor.save(out_stem, config, steps_per_epoch)
    plot_execution(executor._samples, out_stem)

