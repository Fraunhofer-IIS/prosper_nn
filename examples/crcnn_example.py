# %% Package Imports
import sys, os

sys.path.append(os.path.abspath("."))
sys.path.append(os.path.abspath(".."))

import torch
import torch.nn as nn
import torch.optim as optim

import prosper_nn.utils.generate_time_series_data as gtsd
import prosper_nn.utils.create_input_ecnn_hcnn as ci

import prosper_nn.models.crcnn.crcnn as crcnn
from prosper_nn.utils import visualize_forecasts
from matplotlib import pyplot as plt

# %% Define network parameters
past_horizon = 45
forecast_horizon = 5
n_features_Y = 2
n_data = 100
n_state_neurons = 5
batchsize = 5
sparsity = 0
n_branches = 9
teacher_forcing = 1

# %% Create data
# generate data with "unknown" variables U
Y, U = gtsd.sample_data(n_data, n_features_Y=n_features_Y - 1, n_features_U=1)
Y = torch.cat((Y, U), 1)

# Only use Y as input for the Crcnn
Y_batches = ci.create_input(Y, past_horizon, batchsize)
# %% Train model without mirroring and one batch

crcnn_model1 = crcnn.CRCNN(
    n_state_neurons,
    n_features_Y,
    past_horizon,
    forecast_horizon,
    n_branches,
    1,
    sparsity,
    teacher_forcing=teacher_forcing,
    mirroring=False,
)

targets_past = torch.zeros((n_branches - 1, past_horizon, 1, n_features_Y))
optimizer = optim.Adam(crcnn_model1.parameters(), lr=0.001)
loss_function = nn.MSELoss()

epochs = 10
total_loss = epochs * [0]
for epoch in range(epochs):
    crcnn_model1.zero_grad()

    Y_train = Y[:past_horizon].unsqueeze(1)

    model_output = crcnn_model1(Y_train)
    past_error, forecast = torch.split(model_output, past_horizon, dim=1)

    losses_past = [
        loss_function(past_error[i][j], targets_past[i][j])
        for i in range(n_branches - 1)
        for j in range(past_horizon)
    ]
    loss = sum(losses_past) / len(losses_past)
    loss.backward()
    optimizer.step()
    total_loss[epoch] += loss.detach()

plt.plot(total_loss)

# %% Train model with mirroring and one batch

crcnn_model2 = crcnn.CRCNN(
    n_state_neurons,
    n_features_Y,
    past_horizon,
    forecast_horizon,
    n_branches,
    1,
    sparsity,
    teacher_forcing=teacher_forcing,
    mirroring=True,
)
optimizer = optim.Adam(crcnn_model2.parameters(), lr=0.001)
loss_function = nn.MSELoss()

targets_past = torch.zeros((n_branches - 1, past_horizon, 1, n_features_Y))
targets_future = torch.zeros((forecast_horizon, 1, n_features_Y))

epochs = 10
total_loss = epochs * [0]
for epoch in range(epochs):
    crcnn_model2.zero_grad()

    Y_train = Y[:past_horizon].unsqueeze(1)
    model_output = crcnn_model2(Y_train)
    past_error, forecast = torch.split(model_output, past_horizon, dim=1)

    losses_past = [
        loss_function(past_error[i][j], targets_past[i][j])
        for i in range(n_branches - 1)
        for j in range(past_horizon)
    ]
    losses_mirror = [
        loss_function(forecast[-1][j], targets_future[j])
        for j in range(forecast_horizon)
    ]
    loss = sum(losses_mirror) / len(losses_mirror) + sum(losses_past) / len(losses_past)
    loss.backward()
    optimizer.step()
    total_loss[epoch] += loss.detach()

plt.plot(total_loss)

# %% Train model with mirroring and multiple batches

crcnn_model3 = crcnn.CRCNN(
    n_state_neurons,
    n_features_Y,
    past_horizon,
    forecast_horizon,
    n_branches,
    batchsize,
    sparsity,
    teacher_forcing=teacher_forcing,
    mirroring=True,
)

targets_past = torch.zeros((n_branches - 1, past_horizon, batchsize, n_features_Y))
targets_future = torch.zeros((forecast_horizon, batchsize, n_features_Y))
optimizer = optim.Adam(crcnn_model3.parameters(), lr=0.001)
loss_function = nn.MSELoss()
epochs = 10
total_loss = epochs * [0]

# the bias is saved, because it differs for each input, but should be learned across epochs
Bias = torch.zeros((Y_batches.shape[0], forecast_horizon, batchsize, n_features_Y))

for epoch in range(epochs):
    for batch_index in range(0, Y_batches.shape[0]):
        crcnn_model3.zero_grad()

        Y_batch = Y_batches[batch_index]
        if epoch == 0:
            crcnn_model3.future_bias.data = torch.zeros(
                size=(forecast_horizon, batchsize, n_features_Y)
            )
        else:
            crcnn_model3.future_bias.data = Bias[batch_index]
        model_output = crcnn_model3(Y_batch)
        past_error, forecast = torch.split(model_output, past_horizon, dim=1)

        losses_past = [
            loss_function(past_error[i][j], targets_past[i][j])
            for i in range(n_branches - 1)
            for j in range(past_horizon)
        ]
        losses_mirror = [
            loss_function(forecast[-1][j], targets_future[j])
            for j in range(forecast_horizon)
        ]
        loss = sum(losses_mirror) / len(losses_mirror) + sum(losses_past) / len(
            losses_past
        )
        loss.backward()
        optimizer.step()

        total_loss[epoch] += loss.detach()
        Bias[batch_index] = crcnn_model3.future_bias.data.detach()

plt.plot(total_loss)

# %% Evaluation
with torch.no_grad():
    # model 1; Set parameters for evaluation
    crcnn_model1.eval()
    crcnn_model1.CRCNNCell_causal.set_teacher_forcing(1)
    crcnn_model1.CRCNNCell_retro_causal.set_teacher_forcing(1)

    model_output1 = crcnn_model1(Y[:past_horizon].unsqueeze(dim=1))
    past_error1, forecast1 = torch.split(model_output1, past_horizon, dim=1)
    forecast1 = forecast1[-1]

    expected_timeseries1 = torch.cat(
        (torch.add(past_error1[-1].squeeze(), Y[:past_horizon]), forecast1.squeeze()),
        dim=0,
    ).detach()
    for y in range(n_features_Y):
        visualize_forecasts.plot_time_series(
            expected_time_series=expected_timeseries1[:, y],
            target=Y[: past_horizon + forecast_horizon, y],
        )

    # model 2; Set parameters for evaluation
    crcnn_model2.eval()
    crcnn_model2.CRCNNCell_causal.set_teacher_forcing(1)
    crcnn_model2.CRCNNCell_retro_causal.set_teacher_forcing(1)
    crcnn_model2.mirroring = False

    model_output2 = crcnn_model2(Y[:past_horizon].unsqueeze(dim=1))
    past_error2, forecast2 = torch.split(model_output2, past_horizon, dim=1)
    forecast2 = forecast2[-1]

    expected_timeseries2 = torch.cat(
        (torch.add(past_error2[-1].squeeze(), Y[:past_horizon]), forecast2.squeeze()),
        dim=0,
    ).detach()
    for y in range(n_features_Y):
        visualize_forecasts.plot_time_series(
            expected_time_series=expected_timeseries2[:, y],
            target=Y[: past_horizon + forecast_horizon, y],
        )

    # model 3; Set parameters for evaluation

    crcnn_model3.eval()
    crcnn_model3.batchsize = 1
    crcnn_model3.CRCNNCell_causal.batchsize = 1
    crcnn_model3.CRCNNCell_retro_causal.batchsize = 1
    crcnn_model3.CRCNNCell_causal.set_teacher_forcing(1)
    crcnn_model3.CRCNNCell_retro_causal.set_teacher_forcing(1)

    crcnn_model3.mirroring = False
    model_output3 = crcnn_model3(Y[:past_horizon].unsqueeze(dim=1))
    past_error3, forecast3 = torch.split(model_output3, past_horizon, dim=1)
    forecast3 = forecast3[-1]

    expected_timeseries3 = torch.cat(
        (torch.add(past_error3[-1].squeeze(), Y[:past_horizon]), forecast3.squeeze()),
        dim=0,
    ).detach()
    for y in range(n_features_Y):
        visualize_forecasts.plot_time_series(
            expected_time_series=expected_timeseries3[:, y],
            target=Y[: past_horizon + forecast_horizon, y],
        )
