import numpy as np

def aggregate_intervals(step_energies: np.ndarray, step_times: np.ndarray, num_blocks: int = 200):
    """
    Aggregates step energies and times into a fixed number of blocks.
    This reduces the amount of data sent to the UI while preserving the overall energy and time.
    """
    total_steps = len(step_energies)
    if total_steps <= num_blocks:
        return step_energies, step_times

    # Calculate the size of each block
    block_size = total_steps / num_blocks
    
    agg_energies = np.zeros(num_blocks)
    agg_times = np.zeros(num_blocks)
    
    for i in range(num_blocks):
        start_idx = int(i * block_size)
        end_idx = int((i + 1) * block_size)
        
        # Ensure the last block goes all the way to the end
        if i == num_blocks - 1:
            end_idx = total_steps
            
        agg_energies[i] = np.sum(step_energies[start_idx:end_idx])
        agg_times[i] = np.sum(step_times[start_idx:end_idx])
        
    return agg_energies, agg_times

class PowerTranslator:
    """
    Utility class to translate energy per interval into power at any given time.
    Useful for UI visualizations to show power output (Watts) over time.
    """
    def __init__(self, output: dict):
        self.step_energies = np.array(output["step_energy_J"])
        self.step_times = np.array(output["step_time_s"])
        self.step_power = self.step_energies / self.step_times
        self.cum_times = np.cumsum(self.step_times)
        self.total_time = self.cum_times[-1]

    def power_at(self, t: float) -> float:
        """Get the power output (Watts) at a specific time (seconds)."""
        if t < 0 or t > self.total_time:
            return 0.0
        idx = np.searchsorted(self.cum_times, t)
        if idx >= len(self.step_power):
            return 0.0
        return float(self.step_power[idx])

    def to_timeseries(self, resolution_s: float = 1.0) -> dict:
        """Generate a timeseries of power output with a fixed time resolution."""
        times = np.arange(0, self.total_time, resolution_s)
        powers = [self.power_at(t) for t in times]
        return {
            "time_s": times.tolist(),
            "power_W": powers
        }
        
    def get_intervals(self) -> dict:
        """Get the exact intervals with their constant power output."""
        start_times = np.concatenate(([0], self.cum_times[:-1]))
        return {
            "start_time_s": start_times.tolist(),
            "end_time_s": self.cum_times.tolist(),
            "power_W": self.step_power.tolist()
        }
