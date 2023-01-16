# %% Package Imports
import sys, os
sys.path.append(os.path.abspath('..'))
sys.path.append(os.path.abspath('.'))

import torch
import torch.nn as nn
import torch.optim as optim

import prosper_nn.models.dhcnn as dhcnn
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
teacher_forcing = 1
decrease_teacher_forcing = 0.0001

# %% Create data and targets
targets = torch.zeros((past_horizon, batchsize, n_features_Y))
Y, U = gtsd.sample_data(n_data, n_features_Y=n_features_Y-1, n_features_U=1)
Y = torch.cat((Y, U), 1)

# Only use Y as input for the Dhcnn
Y_batches = ci.create_input(Y, past_horizon, batchsize)

# %% Initialize Dhcnn
dhcnn_model = dhcnn.DHCNN(n_state_neurons,
                          n_features_Y,
                          past_horizon,
                          forecast_horizon,
                          deepness,
                          sparsity,
                          activation=torch.tanh,
                          teacher_forcing=teacher_forcing,
                          decrease_teacher_forcing=decrease_teacher_forcing)

# %% Train model
optimizer = optim.Adam(dhcnn_model.parameters(), lr=0.001)
loss_function = nn.MSELoss()

epochs = 1

total_loss = epochs * [0]
for epoch in range(epochs):
    for batch_index in range(Y_batches.shape[0]):
        dhcnn_model.zero_grad()

        Y_batch = Y_batches[batch_index]
        output = dhcnn_model(Y_batch)
        past_error, forecast = torch.split(output, past_horizon, dim=1)

        losses = [loss_function(past_error[i][j], targets[j]) for i in range(deepness) for j in range(past_horizon)]
        loss = sum(losses) / (deepness * past_horizon)
        loss.backward()
        optimizer.step()

        total_loss[epoch] += loss.detach()

# %% Evaluation
# Visualize expected timeseries
expected_timeseries = torch.cat((torch.add(past_error[-1], Y_batches[-1, :past_horizon]),
                                forecast[-1]), dim=0).detach()
visualize_forecasts.plot_time_series(expected_timeseries[:,0,0])
# %%
