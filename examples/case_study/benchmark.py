# %%
import sys
import os

sys.path.append(os.path.abspath("../.."))
sys.path.append(os.path.abspath("../../.."))
sys.path.append(os.path.abspath("...."))
sys.path.append(os.path.abspath("..."))
sys.path.append(os.path.abspath(".."))
sys.path.append(os.path.abspath("."))

# %%
import torch
import pandas as pd
from pathlib import Path

from training import EarlyStopping, Trainer
from evaluation import Evaluator
from models import init_models, Naive
from fredmd import init_datasets

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
    area,
)

# %%
torch.manual_seed(0)


# %% Get Data
fredmd_train, fredmd_val, fredmd_test = init_datasets(
    past_horizon, forecast_horizon, train_test_split_period, area
)

# %% Run benchmark
n_features_U = len(fredmd_train.features)
n_state_neurons = 2 * (n_features_U + n_features_Y)

overall_losses = {}

if n_evaluate_targets == None:
    n_evaluate_targets = len(fredmd_train.features)

benchmark_models = {}
for index_target, target in enumerate(fredmd_train.features[:n_evaluate_targets]):
    fredmd_train.target = target
    fredmd_val.target = target
    fredmd_test.target = target

    benchmark_models = init_models(
        benchmark_models,
        n_features_U,
        n_state_neurons,
        n_features_Y,
        past_horizon,
        forecast_horizon,
        n_models,
    )

    for name, model in benchmark_models.items():
        is_untrained_multivariate_model = model.multivariate and (index_target == 0)
        not_naive_model = not isinstance(model, Naive)

        needs_training = (
            not model.multivariate or is_untrained_multivariate_model
        ) and not_naive_model
        if needs_training:
            fredmd_train.set_target_future_format(multivariate=model.multivariate)
            fredmd_val.set_target_future_format(multivariate=model.multivariate)

            dataloader = torch.utils.data.DataLoader(
                fredmd_train, batch_size=batch_size, shuffle=True, drop_last=True
            )
            trainer = Trainer(model, EarlyStopping(patience), n_epochs)
            print(f"### Train {name} ###")
            trainer.train(dataloader, fredmd_val)

    if target == "DNDGRG3M086SBEA":
        torch.save(
            benchmark_models["ECNN"], Path(__file__).parent / f"ECNN_{target}.pt"
        )

    # Test
    losses_one_target = {}
    for name, model in benchmark_models.items():
        fredmd_test.set_target_future_format(multivariate=False)
        evaluator = Evaluator(model, forecast_horizon)
        loss_one_target_one_model = evaluator.evaluate(fredmd_test, index_target)
        losses_one_target[name] = loss_one_target_one_model

    overall_losses[target] = pd.concat(losses_one_target)

overall_losses = pd.concat(overall_losses)
overall_losses.to_csv(Path(__file__).parent / f"overall_losses_{area}.csv")
mean_overall_losses = overall_losses.groupby(level=1).mean()
mean_overall_losses.to_csv(Path(__file__).parent / f"mean_overall_losses_{area}.csv")
print(mean_overall_losses)
