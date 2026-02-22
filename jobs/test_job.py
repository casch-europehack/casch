import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from profiler.models import JobConfig, TrainingJob


class MyTrainingJob(TrainingJob):
    def configure(self) -> JobConfig:
        return JobConfig(
            total_epochs=24,
            gpu_indices=[0],
            profile_steps=50,
        )

    def setup(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = nn.Sequential(
            nn.Linear(784, 5120),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(5120, 5120),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(5120, 5120),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(5120, 5120),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(5120, 5120),
            nn.ReLU(),
            nn.Linear(5120, 10),
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.loss_fn = nn.CrossEntropyLoss()

        X = torch.randn(1_200_000, 784)
        y = torch.randint(0, 10, (1_200_000,))
        self.dataset = TensorDataset(X, y)

    def get_dataloader(self) -> DataLoader:
        return DataLoader(self.dataset, batch_size=256, shuffle=True, num_workers=0)

    def train_one_step(self, batch) -> None:
        X, y = batch
        X, y = X.to(self.device), y.to(self.device)
        self.optimizer.zero_grad()
        loss = self.loss_fn(self.model(X), y)
        loss.backward()
        self.optimizer.step()
