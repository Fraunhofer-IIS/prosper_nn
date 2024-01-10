import torch

import prosper_nn.utils.generate_time_series_data as gtsd
import prosper_nn.utils.create_input_ecnn_hcnn as ci
from prosper_nn.models.ecnn import ECNN

# Define network and data parameters
past_horizon = 10
forecast_horizon = 5
n_features_U = 2
n_features_Y = 2
n_data = 20
n_state_neurons = 4
batchsize = 1

# Initialise Error Correction Neural Network
ecnn = ECNN(
    n_features_U,
    n_state_neurons,
    past_horizon,
    forecast_horizon,
    n_features_Y=n_features_Y,
)

# Generate data
Y, U = gtsd.sample_data(n_data, n_features_Y, n_features_U)
Y_batches, U_batches = ci.create_input(
    Y=Y,
    past_horizon=past_horizon,
    batchsize=batchsize,
    U=U,
    forecast_horizon=forecast_horizon,
)

targets = torch.zeros((past_horizon, batchsize, n_features_Y))

# Train model
optimizer = torch.optim.Adam(ecnn.parameters())
loss_function = torch.nn.MSELoss()

for epoch in range(10):
    for batch_index in range(0, U_batches.shape[0]):
        U_batch = U_batches[batch_index]
        Y_batch = Y_batches[batch_index]
        model_output = ecnn(U_batch, Y_batch)
        past_error, forecast = torch.split(model_output, past_horizon)

        ecnn.zero_grad()
        loss = loss_function(past_error, targets)
        loss.backward()
        optimizer.step()
