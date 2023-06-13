# %% Package Imports
import sys, os

sys.path.append(os.path.abspath(".."))
sys.path.append(os.path.abspath("."))

import torch
import torch.nn as nn
import torch.optim as optim

import prosper_nn.models.crcnn as crcnn
import prosper_nn.models.ensemble as ensemble
from prosper_nn.utils import visualize_forecasts
import prosper_nn.utils.generate_time_series_data as gtsd
import prosper_nn.utils.create_input_ecnn_hcnn as ci

# %% Define network parameters
past_horizon = 20
forecast_horizon = 5
n_features_Y = 1
n_data = 100
n_state_neurons = 3
batchsize = 2
sparsity = 0
n_branches = 7
n_models = 2

# %% Create data and targets
targets = torch.zeros((n_branches - 1, past_horizon, batchsize, n_features_Y))

# generate data with "unknown" variables U
n_features_U = 5
Y, _ = gtsd.sample_data(n_data, n_features_Y=n_features_Y, n_features_U=n_features_U)
# Only use Y as input for the Crcnn
Y_batches = ci.create_input(Y, past_horizon, batchsize)

# %% Initialize Crcnn and an ensemble of it
crcnn_model = crcnn.CRCNN(
    n_state_neurons,
    n_features_Y,
    past_horizon,
    forecast_horizon,
    n_branches,
    batchsize,
    sparsity,
    mirroring=False,
)

crcnn_ensemble = ensemble.Ensemble(
    model=crcnn_model,
    n_models=n_models,
    sparsity=sparsity,
    keep_pruning_mask=False,
    initializer=torch.nn.init.kaiming_uniform_,
)

# %% Train model
optimizer = optim.Adam(crcnn_ensemble.parameters(), lr=0.001)
loss_function = nn.MSELoss()

epochs = 2

total_loss = epochs * [0]
for epoch in range(epochs):
    for batch_index in range(0, Y_batches.shape[0]):
        crcnn_ensemble.zero_grad()

        Y_batch = Y_batches[batch_index]
        ensemble_output = crcnn_ensemble(Y_batch)
        outputs, mean = torch.split(ensemble_output, n_models)
        past_errors, forecasts = torch.split(outputs, past_horizon, dim=2)

        losses = [
            loss_function(past_errors[j][k][i], targets[k][i])
            for i in range(past_horizon)
            for k in range(n_branches - 1)
            for j in range(n_models)
        ]
        loss = sum(losses) / (n_models * n_branches * past_horizon)
        loss.backward()
        optimizer.step()

        mean_errors = torch.mean(past_errors[:, -1], dim=0)
        mean_forecasts = torch.mean(forecasts[:, -1], dim=0)
        mean_targets = targets[0]

        mean_loss = sum(
            [
                loss_function(mean_errors[i], mean_targets[i])
                for i in range(past_horizon)
            ]
        )
        total_loss[epoch] += mean_loss.detach()

# %% Evaluation
# Visualization of the expected timeseries
expected_timeseries = torch.cat(
    (torch.add(mean_errors, Y_batches[-1, :past_horizon]), mean_forecasts), dim=0
).detach()
expected_timeseries_outputs = torch.cat(
    (torch.add(past_errors[:, -1], Y_batches[-1, :past_horizon]), forecasts[:, -1]),
    dim=1,
).detach()

visualize_forecasts.plot_time_series(
    expected_time_series=expected_timeseries[:, 0, 0],
    target=Y[-(past_horizon + forecast_horizon) :],
    uncertainty=expected_timeseries_outputs[:, :, 0, 0].T,
)
visualize_forecasts.heatmap_forecasts(expected_timeseries_outputs[:, :, 0, 0].T)
