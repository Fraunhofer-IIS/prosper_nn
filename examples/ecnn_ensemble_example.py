import os
import sys
import inspect

current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

import torch
import torch.nn as nn
import torch.optim as optim

import prosper_nn.utils.generate_time_series_data as gtsd
import prosper_nn.utils.create_input_ecnn_hcnn as ci
import prosper_nn.models.ecnn as ecnn
import prosper_nn.models.ensemble as ensemble

# %% Define network parameters

n_features_U = 3
future_U = False
n_state_neurons = 3
batchsize = 1
past_horizon = 8
forecast_horizon = 4
init_state = torch.zeros(1, n_state_neurons)

n_data = 100
n_features_Y = 2
n_models = 3

# %% Generate dummy data
Y, U = gtsd.sample_data(n_data, n_features_Y, n_features_U)

Y_batches, U_batches = ci.create_input(
    Y, past_horizon, batchsize, U, future_U, forecast_horizon
)

targets = torch.zeros((past_horizon, batchsize, n_features_Y))

# Initialize Ecnn and an ensemble of it
ecnn = ecnn.ECNN(
    n_features_U=n_features_U,
    n_state_neurons=n_state_neurons,
    past_horizon=past_horizon,
    forecast_horizon=forecast_horizon,
    approach="backward",
    init_state=init_state,
    learn_init_state=True,
    n_features_Y=n_features_Y,
    future_U=future_U,
)

ensemble_model = ensemble.Ensemble(
    model=ecnn, n_models=n_models, initializer=torch.nn.init.kaiming_uniform_
)

# %% Train model

optimizer = optim.Adam(ensemble_model.parameters(), lr=0.001)
loss_function = nn.MSELoss()

epochs = 10
total_loss = epochs * [0]
for epoch in range(epochs):
    for batch_index in range(0, U_batches.size(0)):
        ensemble_model.zero_grad()

        U_batch = U_batches[batch_index]
        Y_batch = Y_batches[batch_index]

        ensemble_output = ensemble_model(U_batch, Y_batch)
        outputs, mean = torch.split(ensemble_output, n_models)
        mean = torch.squeeze(mean, 0)
        past_errors, forecasts = torch.split(outputs, past_horizon, dim=1)

        losses = [
            loss_function(past_errors[j][i], targets[i])
            for j in range(n_models)
            for i in range(past_horizon)
        ]
        loss = sum(losses) / (n_models * past_horizon)
        loss.backward()
        optimizer.step()

        mean_loss = (
            sum([loss_function(mean[i], targets[i]) for i in range(past_horizon)])
            / past_horizon
        )
        total_loss[epoch] += mean_loss.detach()

# %% Prediction
# example data for prediction
if future_U:
    example_pred_U = U[: (past_horizon + forecast_horizon)].unsqueeze(dim=1)
else:
    example_pred_U = U[:past_horizon].unsqueeze(dim=1)
example_pred_Y = Y[: (past_horizon + forecast_horizon)].unsqueeze(dim=1)

# predict with trained model
with torch.no_grad():
    ensemble_model.eval()

    ensemble_output = ensemble_model(example_pred_U, example_pred_Y[:past_horizon])
    _, ensemble_forecast = torch.split(ensemble_output, past_horizon, dim=1)

    mean_forecast = ensemble_forecast[-1]
