import sys, os

sys.path.append(os.path.abspath(".."))
sys.path.append(os.path.abspath("."))

import torch
import torch.nn as nn
import torch.optim as optim

import prosper_nn.utils.generate_time_series_data as gtsd
import prosper_nn.utils.create_input_ecnn_hcnn as ci
import prosper_nn.models.ecnn as ecnn
from prosper_nn.utils import visualize_forecasts
import prosper_nn.utils.neuron_correlation_hidden_layers as nchl

# %% Define network parameters
n_features_U = 2
batchsize = 5
past_horizon = 10
forecast_horizon = 2
future_U = False
n_state_neurons = 5

n_data = 1500
n_features_Y = 2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %% Generate data and targets
Y, U = gtsd.sample_data(n_data, n_features_Y, n_features_U)
Y_batches, U_batches = ci.create_input(
    Y, past_horizon, batchsize, U, future_U, forecast_horizon
)
Y_batches = Y_batches.to(device)
U_batches = U_batches.to(device)

targets = torch.zeros((past_horizon, batchsize, n_features_Y), device=device)

# %% Initialize ECNN
ecnn_model = ecnn.ECNN(
    n_features_U,
    n_state_neurons,
    past_horizon,
    forecast_horizon,
    lstm=False,
    approach="backward",
    learn_init_state=True,
    n_features_Y=n_features_Y,
    future_U=future_U,
).to(device)

# %% Train model
optimizer = optim.Adam(ecnn_model.parameters(), lr=0.01)
loss_function = nn.MSELoss()

epochs = 10

total_loss = epochs * [0]
for epoch in range(epochs):
    for batch_index in range(0, U_batches.shape[0]):
        ecnn_model.zero_grad()

        U_batch = U_batches[batch_index]
        Y_batch = Y_batches[batch_index]
        model_output = ecnn_model(U_batch, Y_batch)
        past_error, forecast = torch.split(model_output, past_horizon)

        loss = loss_function(past_error, targets)
        loss.backward()
        optimizer.step()
        total_loss[epoch] += loss.detach()

# %% Prediction
# Example data for prediction

if future_U:
    example_pred_U = torch.reshape(
        U[0 : (past_horizon + forecast_horizon), :],
        (past_horizon + forecast_horizon, 1, n_features_U),
    ).float()
else:
    example_pred_U = torch.reshape(
        U[0:past_horizon, :], (past_horizon, 1, n_features_U)
    ).float()
example_pred_Y = torch.reshape(
    Y[0 : (past_horizon + forecast_horizon), :],
    (past_horizon + forecast_horizon, 1, n_features_Y),
).float()

# Predict with trained model
with torch.no_grad():
    ecnn_model.eval()

    model_output = ecnn_model(example_pred_U.to(device), example_pred_Y[:past_horizon].to(device))
    past_errors, forecast = torch.split(model_output, past_horizon)
    print("Forecast: {}".format(forecast))
    expected_timeseries = (
        torch.cat(
            (torch.add(past_errors.cpu(), example_pred_Y[:past_horizon]), forecast.cpu()), dim=0
        )
        .detach()
        .squeeze()
    )

    if ecnn_model.lstm:
        print(ecnn_model.ecnn_cell.D.weight.data)

    visualize_forecasts.plot_time_series(
        expected_timeseries, example_pred_Y.squeeze(1)[:, 0]
    )

    # neuron correlation analysis to check size of hidden layer (= n_state_neurons)
    states_for_correlation = torch.empty(
        (U_batches.shape[0], batchsize, n_state_neurons), device=device
    )
    for batch_index in range(0, U_batches.shape[0]):
        U_batch = U_batches[batch_index]
        Y_batch = Y_batches[batch_index]
        model_output = ecnn_model(U_batch, Y_batch)
        states_for_correlation[batch_index, :, :] = ecnn_model.state[past_horizon]
    states_for_correlation = states_for_correlation.reshape((-1, n_state_neurons))
    corr_matrix, max_corr, ind_neurons = nchl.hl_size_analysis(states_for_correlation.cpu())
