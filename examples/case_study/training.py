import torch
from torch import nn
from tqdm import tqdm
from copy import deepcopy


class EarlyStopping:
    def __init__(self, patience):
        self.patience = patience
        self.smallest_val_loss = torch.inf
        self.epoch_smallest_val = 0
        self.best_model_state = None

    def early_stop(self, epoch):
        stop = epoch >= self.epoch_smallest_val + self.patience
        if stop:
            print(
                f"No validation improvement since {self.patience} epochs -> Stop Training"
            )
        return stop

    def set_best_model_state(self, model, epoch, val_loss):
        if val_loss < self.smallest_val_loss:
            print(f"Save model_state at epoch {epoch}")
            self.best_model_state = deepcopy(model.state_dict())
            self.smallest_val_loss = val_loss
            self.epoch_smallest_val = epoch


class Trainer:
    def __init__(self, model: nn.Module, early_stopping: EarlyStopping, n_epochs: int):
        self.model = model
        self.early_stopping = early_stopping
        self.train_loss = 0
        self.epochs = tqdm(range(n_epochs))
        self.optimizer = torch.optim.Adam(self.model.parameters())

    def train(
        self,
        dataloader: torch.utils.data.DataLoader,
        dataset_val: torch.utils.data.Dataset,
    ):
        val_features_past, val_target_past, val_target_future = (
            dataset_val.get_all_rolling_origins()
        )

        for epoch in self.epochs:
            self.train_loss = 0
            for features_past, target_past, target_future in dataloader:
                target_past = target_past.transpose(1, 0)
                target_future = target_future.transpose(1, 0)
                features_past = features_past.transpose(1, 0)

                self.model.zero_grad()

                input = self.model.get_input(features_past, target_past)
                output = self.model(*input)
                forecasts = self.model.extract_forecasts(output)
                assert forecasts.shape == target_future.shape
                loss = nn.functional.mse_loss(forecasts, target_future)
                loss.backward()
                self.train_loss += loss.detach()
                self.optimizer.step()

            # Validation loss
            val_loss = self.get_validation_loss(
                val_features_past, val_target_past, val_target_future
            )
            self.epochs.set_postfix(
                {
                    "val_loss": round(val_loss, 3),
                    "train_loss": round(self.train_loss.item(), 3),
                }
            )

            self.early_stopping.set_best_model_state(self.model, epoch, val_loss)
            if self.early_stopping.early_stop(epoch):
                self.model.load_state_dict(self.early_stopping.best_model_state)
                return

        self.model.load_state_dict(self.early_stopping.best_model_state)

    def get_validation_loss(
        self, val_features_past, val_target_past, val_target_future
    ):
        input = self.model.get_input(val_features_past, val_target_past)
        output = self.model(*input)
        forecasts_val = self.model.extract_forecasts(output)
        val_loss = nn.functional.mse_loss(forecasts_val[0], val_target_future[0]).item()
        return val_loss
