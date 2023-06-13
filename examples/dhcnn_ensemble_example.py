# %% Package Imports
import sys, os

sys.path.append(os.path.abspath(".."))
sys.path.append(os.path.abspath("."))

import torch
import torch.nn as nn
import torch.optim as optim

import prosper_nn.models.dhcnn as dhcnn
import prosper_nn.models.ensemble as ensemble
from prosper_nn.utils import visualize_forecasts

import prosper_nn.utils.generate_time_series_data as gtsd
import prosper_nn.utils.create_input_ecnn_hcnn as ci

# %% Define network parameters
past_horizon = 20
forecast_horizon = 10
n_features_Y = 2
n_data = 100
n_state_neurons = 3
batchsize = 2
n_batches = 2
sparsity = 0
deepness = 3
n_models = 3
# %% Create data and targets

targets = torch.zeros((past_horizon, batchsize, n_features_Y))
Y, U = gtsd.sample_data(n_data, n_features_Y=n_features_Y - 1, n_features_U=1)
Y = torch.cat((Y, U), 1)

# Only use Y as input for the Dhcnn
Y_batches = ci.create_input(Y, past_horizon, batchsize)

#%% Initialize Dhcnn model and an ensemble of it
dhcnn_model = dhcnn.DHCNN(
    n_state_neurons,
    n_features_Y,
    past_horizon,
    forecast_horizon,
    deepness,
    sparsity,
    activation=torch.tanh,
)

dhcnn_ensemble = ensemble.Ensemble(
    model=dhcnn_model,
    n_models=n_models,
    sparsity=sparsity,
    keep_pruning_mask=False,
    initializer=torch.nn.init.kaiming_uniform_,
)

# %% Train model
optimizer1 = optim.Adam(dhcnn_ensemble.parameters(), lr=0.001)
loss_function = nn.MSELoss()

epochs = 10

total_loss = epochs * [0]
for epoch in range(epochs):
    for batch_index in range(Y_batches.shape[0]):
        dhcnn_ensemble.zero_grad()

        Y_batch = Y_batches[batch_index]
        ensemble_output = dhcnn_ensemble(Y_batch)
        outputs, mean = torch.split(ensemble_output, n_models)
        past_errors, forecasts = torch.split(outputs, past_horizon, dim=2)
        mean = torch.squeeze(mean, 0)

        losses = [
            loss_function(past_errors[k][i][j], targets[j])
            for i in range(deepness)
            for j in range(past_horizon)
            for k in range(n_models)
        ]
        loss = sum(losses) / (deepness * past_horizon * n_models)
        loss.backward()
        optimizer1.step()

        mean_loss = (
            sum([loss_function(mean[-1, i], targets[i]) for i in range(past_horizon)])
            / past_horizon
        )
        total_loss[epoch] += mean_loss.detach()

# %% Evaluation
# Visualize expected timeseries
expected_timeseries = torch.cat(
    (
        torch.add(mean[-1, :past_horizon], Y_batches[-1, :past_horizon]),
        mean[-1, past_horizon:],
    ),
    dim=0,
).detach()
visualize_forecasts.plot_time_series(expected_timeseries[:, 0, 0])
