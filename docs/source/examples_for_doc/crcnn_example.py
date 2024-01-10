import torch

from prosper_nn.models.crcnn import CRCNN
import prosper_nn.utils.generate_time_series_data as gtsd
import prosper_nn.utils.create_input_ecnn_hcnn as ci

# Define network and data parameters
past_horizon = 10
forecast_horizon = 5
n_features_Y = 2
n_data = 20
n_state_neurons = 3
n_branches = 3
batchsize = 5

# Initialise Causal-Retro-Causal Neural Network
crcnn_model = CRCNN(
    n_state_neurons, n_features_Y, past_horizon, forecast_horizon, n_branches
)

# Generate data with "unknown" variables U
Y, U = gtsd.sample_data(n_data, n_features_Y=n_features_Y - 1, n_features_U=1)
Y = torch.cat((Y, U), 1)
Y_batches = ci.create_input(Y, past_horizon, batchsize)

targets = torch.zeros((n_branches - 1, past_horizon, batchsize, n_features_Y))

# Train model
optimizer = torch.optim.Adam(crcnn_model.parameters())
loss_function = torch.nn.MSELoss()

for epoch in range(10):
    for batch_index in range(0, Y_batches.shape[0]):
        Y_batch = Y_batches[batch_index]
        model_output = crcnn_model(Y_batch)
        past_errors, forecasts = torch.split(model_output, past_horizon, dim=1)

        crcnn_model.zero_grad()
        loss = loss_function(past_errors, targets)
        loss.backward()
        optimizer.step()
