from abc import ABC, abstractmethod
from dataclasses import dataclass
from torch.utils.data import DataLoader


@dataclass
class JobConfig:
    """
    Configuration for a training job
    that can be profiled.
    """

    total_epochs: int
    gpu_indices: list[int]
    profile_steps: int = 50

class TrainingJob(ABC):
    """
    Abstract base class for training jobs that
    can be profiled.
    """

    @abstractmethod
    def configure(self) -> JobConfig:
        ...

    @abstractmethod
    def setup(self) -> None:
        ...

    @abstractmethod
    def get_dataloader(self) -> DataLoader:
        ...

    @abstractmethod
    def train_one_step(self, batch) -> None:
        ...

    def teardown(self) -> None:
        pass
