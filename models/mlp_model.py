import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt

from .base_model import BaseModel


class MLP(nn.Module):

    def __init__(self):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(2, 2),
            nn.ReLU(),
            nn.Linear(2, 1)
        )

    def forward(self, x):
        x = self.layers(x)
        return x


class MLPDataset(Dataset):

    def __init__(self, features, labels):
        self.features = features.to_numpy()
        self.labels = labels.to_numpy()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return torch.from_numpy(self.features[idx]).float(), torch.from_numpy(self.labels[idx]).float()


class MLPModel(BaseModel):

    def __init__(self, lr=1e-3):
        super().__init__()
        self.model = MLP()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

    def _train_model(self, train_loader):
        self.model.train()
        train_losses = []

        pbar = tqdm(train_loader)
        for features, labels in pbar:
            self.optimizer.zero_grad()
            outputs = self.model(features)
            loss = self.loss_fn(outputs, labels)
            loss.backward()
            self.optimizer.step()

            train_losses.append(loss.item())
            pbar.set_postfix({"train_loss": np.mean(train_losses)})

        return train_losses

    @torch.no_grad()
    def _eval_model(self, test_loader):
        self.model.eval()
        eval_losses = []

        pbar = tqdm(test_loader)
        for features, labels in pbar:
            outputs = self.model(features)
            loss = self.loss_fn(outputs, labels)

            eval_losses.append(loss.item())
            pbar.set_postfix({"eval_loss": np.mean(eval_losses)})

        return eval_losses

    def fit(self, X_train, y_train, X_test, y_test, batch_size=64, epochs=10):
        train_data = MLPDataset(X_train, y_train)
        test_data = MLPDataset(X_test, y_test)

        train_loader = DataLoader(train_data, batch_size=batch_size)
        test_loader = DataLoader(test_data, batch_size=batch_size)

        self.mean_train_losses = []
        self.mean_valid_losses = []

        for epoch in range(epochs):
            print(f'Epoch {epoch + 1}/{epochs}')

            train_losses = self._train_model(train_loader)
            valid_losses = self._eval_model(test_loader)

            self.mean_train_losses.append(np.mean(train_losses))
            self.mean_valid_losses.append(np.mean(valid_losses))

    def plot_metric(self):
        fig, ax = plt.subplots()
        ax.set_title("Metric during training")
        ax.set_xlabel("Epochs")
        ax.set_ylabel("MSE")

        ax.plot(self.mean_train_losses, label="train")
        ax.plot(self.mean_valid_losses, label="valid")
        ax.legend()
        ax.grid()
        return ax
