# %%
import sys
import os

sys.path.append(os.path.abspath("../.."))
sys.path.append(os.path.abspath(".."))
sys.path.append(os.path.abspath("."))


# %%
import torch
import torch.nn as nn
import pandas as pd
from tqdm import tqdm
from pathlib import Path

from prosper_nn.models.ecnn import ECNN
from prosper_nn.models.ensemble import Ensemble
from models import RNN_direct, RNN_recursive, RNN_S2S, Naive
from fredmd import Dataset_Fredmd

from config import (
    past_horizon,
    forecast_horizon,
    train_test_split_period,
    batch_size,
    n_epochs,
    patience,
    n_evaluate_targets,
    n_features_Y,
    n_models,
)

# %%
torch.manual_seed(0)


# %% Training
def train_model(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    dataset_val: torch.utils.data.Dataset,
    n_epochs: int,
    patience: int,
):
    optimizer = torch.optim.Adam(model.parameters())
    smallest_val_loss = torch.inf
    epoch_smallest_val = 0
    val_features_past, val_target_past, val_target_future = (
        dataset_val.get_all_rolling_origins()
    )
    epochs = tqdm(range(n_epochs))

    for epoch in epochs:
        train_loss = 0
        for features_past, target_past, target_future in dataloader:
            target_past = target_past.transpose(1, 0)
            target_future = target_future.transpose(1, 0)
            features_past = features_past.transpose(1, 0)

            model.zero_grad()

            forecasts = get_forecast(model, features_past, target_past)

            assert forecasts.shape == target_future.shape
            loss = nn.functional.mse_loss(forecasts, target_future)
            loss.backward()
            train_loss += loss.detach()
            optimizer.step()

        # Validation loss
        forecasts_val = get_forecast(model, val_features_past, val_target_past)
        val_loss = nn.functional.mse_loss(forecasts_val[0], val_target_future[0]).item()
        epochs.set_postfix(
            {"val_loss": round(val_loss, 3), "train_loss": round(train_loss.item(), 3)}
        )

        # Save and later use model with best validation loss
        if val_loss < smallest_val_loss:
            print(f"Save model_state at epoch {epoch}")
            best_model_state = model.state_dict()
            smallest_val_loss = val_loss
            epoch_smallest_val = epoch

        # Early Stopping
        if epoch >= epoch_smallest_val + patience:
            print(f"No validation improvement since {patience} epochs -> Stop Training")
            model.load_state_dict(best_model_state)
            return

    model.load_state_dict(best_model_state)


def get_forecast(
    model: nn.Module, features_past: torch.Tensor, target_past: torch.Tensor
) -> torch.Tensor:
    model_type = model.models[0]

    # Select input
    if isinstance(model_type, ECNN):
        input = (features_past, target_past)
    else:
        input = (features_past,)

    ensemble_output = model(*input)
    mean = ensemble_output[-1]

    # Extract forecasts
    if isinstance(model_type, ECNN):
        _, forecasts = torch.split(mean, past_horizon)
    else:
        forecasts = mean
    return forecasts


def evaluate_model(model: nn.Module, dataset: torch.utils.data.Dataset) -> pd.DataFrame:
    model.eval()
    losses = []

    for features_past, target_past, target_future in dataset:
        features_past = features_past.unsqueeze(1)
        target_past = target_past.unsqueeze(1)

        with torch.no_grad():
            forecasts = get_forecast(model, features_past, target_past)
        forecasts = forecasts.squeeze(1)
        assert forecasts.shape == target_future.shape
        losses.append(
            [
                nn.functional.mse_loss(forecasts[i], target_future[i]).item()
                for i in range(forecast_horizon)
            ]
        )
    return pd.DataFrame(losses)


# %% Get Data

fredmd = Dataset_Fredmd(
    past_horizon,
    forecast_horizon,
    split_date=train_test_split_period,
    data_type="train",
)
fredmd_val = Dataset_Fredmd(
    past_horizon,
    forecast_horizon,
    split_date=train_test_split_period,
    data_type="val",
)
fredmd_test = Dataset_Fredmd(
    past_horizon,
    forecast_horizon,
    split_date=train_test_split_period,
    data_type="test",
)

# %% Run benchmark
n_features_U = len(fredmd.features)
n_state_neurons = n_features_U + n_features_Y

overall_losses = {}

for target in fredmd.features[:n_evaluate_targets]:
    fredmd.target = target
    fredmd_val.target = target
    fredmd_test.target = target

    # Error Correction Neural Network (ECNN)
    ecnn = ECNN(
        n_state_neurons=n_state_neurons,
        n_features_U=n_features_U,
        n_features_Y=n_features_Y,
        past_horizon=past_horizon,
        forecast_horizon=forecast_horizon,
    )

    # Define an Ensemble for better forecasts, heatmap visualization and sensitivity analysis
    ecnn_ensemble = Ensemble(model=ecnn, n_models=n_models).double()
    benchmark_models = {"ECNN": ecnn_ensemble}

    # Compare to further Recurrent Neural Networks
    for forecast_module in [RNN_direct, RNN_recursive, RNN_S2S]:
        for recurrent_cell_type in ["elman", "gru", "lstm"]:
            model = forecast_module(
                n_features_U,
                n_state_neurons,
                n_features_Y,
                forecast_horizon,
                recurrent_cell_type,
            )
            ensemble = Ensemble(model=model, n_models=n_models).double()
            benchmark_models[f"{recurrent_cell_type}_{model.forecast_method}"] = (
                ensemble
            )

    # Train models
    dataloader = torch.utils.data.DataLoader(
        fredmd, batch_size=batch_size, shuffle=True
    )

    for name, model in benchmark_models.items():
        print(f"### Train {name} ###")
        train_model(model, dataloader, fredmd_val, n_epochs, patience)

    if target == "DNDGRG3M086SBEA":
        torch.save(
            benchmark_models["ECNN"], Path(__file__).parent / f"ECNN_{target}.pt"
        )

    # Test
    # Additionally, compare with the naive no-change forecast
    benchmark_models["Naive"] = Ensemble(
        Naive(past_horizon, forecast_horizon, n_features_Y), n_models
    )

    all_losses = {
        name: evaluate_model(model, fredmd_test)
        for name, model in benchmark_models.items()
    }
    overall_losses[target] = pd.concat(all_losses)

overall_losses = pd.concat(overall_losses)
overall_losses.to_csv(Path(__file__).parent / f"overall_losses.csv")
mean_overall_losses = overall_losses.groupby(level=1).mean()
mean_overall_losses.to_csv(Path(__file__).parent / f"mean_overall_losses.csv")
print(mean_overall_losses)
