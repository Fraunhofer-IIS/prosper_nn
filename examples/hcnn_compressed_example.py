import sys, os

sys.path.append(os.path.abspath(".."))
sys.path.append(os.path.abspath("."))

import torch
import torch.nn as nn
import torch.optim as optim

from prosper_nn.models.hcnn_compressed import HCNN_compressed
import prosper_nn.utils.generate_time_series_data as gtsd
import prosper_nn.utils.create_input_ecnn_hcnn as ci

# %% Define network parameters

n_data = 10
batchsize = 1
n_batches = 2
n_state_neurons = 10
n_features_task = 2
n_features_support = 5
n_features_cSupport = 3
past_horizon = 10
forecast_horizon = 5


# %% Create data and targets
target_task = torch.zeros((past_horizon, batchsize, n_features_task))
target_cSupport = torch.zeros((past_horizon, batchsize, n_features_cSupport))

# generate data with "unknown" variables U
support, task = gtsd.sample_data(
    n_data, n_features_Y=n_features_support, n_features_U=n_features_task
)

# Only use Y as input for the hcnn
batches_task = ci.create_input(task, past_horizon, batchsize)
batches_support = ci.create_input(support, past_horizon, batchsize)

# %% Initialize Hcnn
hcnn_model_compressed = HCNN_compressed(
    n_state_neurons,
    n_features_task,
    n_features_support,
    n_features_cSupport,
    past_horizon,
    forecast_horizon,
)


# %% Train model
optimizer = optim.Adam(hcnn_model_compressed.parameters(), lr=0.001)
loss_function = nn.MSELoss()

epochs = 10

total_loss = epochs * [0]

for epoch in range(epochs):
    for batch_index in range(batches_task.shape[0]):
        hcnn_model_compressed.zero_grad()

        output_task, output_support = hcnn_model_compressed(
            batches_task[batch_index], batches_support[batch_index]
        )

        past_error_task, forecast_task = torch.split(output_task, past_horizon)
        past_error_support = output_support[:past_horizon]

        losses_task = loss_function(past_error_task, target_task)
        losses_support = loss_function(past_error_support, target_cSupport)

        loss = losses_task + losses_support
        loss.backward()
        optimizer.step()
        total_loss[epoch] += loss.detach()
