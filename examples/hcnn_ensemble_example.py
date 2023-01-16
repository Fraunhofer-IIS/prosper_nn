# %% Package Imports
import sys, os
sys.path.append(os.path.abspath('..'))
sys.path.append(os.path.abspath('.'))

import torch
import torch.nn as nn
import torch.optim as optim

import prosper_nn.models.hcnn as hcnn
import prosper_nn.models.ensemble as ensemble
from prosper_nn.utils import sensitivity_analysis
from prosper_nn.utils import visualize_forecasts
import prosper_nn.utils.generate_time_series_data as gtsd
import prosper_nn.utils.create_input_ecnn_hcnn as ci

# %% Define network parameters
past_horizon = 20
forecast_horizon = 10
n_features_Y = 2
n_data = 100
n_state_neurons = 3
batchsize = 1
n_batches = 2
sparsity = 0.2
n_models = 3

# %% Create data and targets
targets = torch.zeros((past_horizon, batchsize, n_features_Y))

# generate data with "unknown" variables U
Y, U = gtsd.sample_data(n_data, n_features_Y=n_features_Y-1, n_features_U=1)
Y = torch.cat((Y, U), 1)

# Only use Y as input for the Hcnn
Y_batches = ci.create_input(Y, past_horizon, batchsize)

# %% Initialize Hcnn and an ensemble of it
hcnn_model = hcnn.HCNN(n_state_neurons,
                       n_features_Y,
                       past_horizon,
                       forecast_horizon,
                       sparsity,
                       lstm=False)

hcnn_ensemble = ensemble.Ensemble(model=hcnn_model,
                                  n_models=n_models,
                                  sparsity=sparsity,
                                  keep_pruning_mask=False,
                                  initializer=torch.nn.init.kaiming_uniform_)

# %% Train model
optimizer1 = optim.Adam(hcnn_ensemble.parameters(), lr=0.001)
loss_function = nn.MSELoss()

epochs = 5

total_loss = epochs * [0]
for epoch in range(epochs):
    for batch_index in range(0, Y_batches.shape[0]):
        hcnn_ensemble.zero_grad()

        Y_batch = Y_batches[batch_index]
        ensemble_output = hcnn_ensemble(Y_batch)
        outputs, mean = torch.split(ensemble_output, n_models)
        mean = torch.squeeze(mean, 0)
        past_errors, forecasts = torch.split(outputs, past_horizon, dim=1)
        
        losses = [loss_function(past_errors[j][i], targets[i]) for j in range(n_models)
                                                                for i in range(past_horizon)]
        loss = sum(losses) / (n_models * past_horizon)
        loss.backward(retain_graph=False)
        optimizer1.step()

        mean_loss = sum([loss_function(mean[i], targets[i]) for i in range(past_horizon)]) / past_horizon
        total_loss[epoch] += mean_loss.detach()

# %% Evaluation
# Visualization of the expected timeseries
expected_timeseries = torch.cat((torch.add(mean[:past_horizon], Y_batches[-1, :past_horizon]),
                                mean[past_horizon:]), dim=0).detach()
expected_timeseries_outputs = torch.cat((torch.add(past_errors, Y_batches[-1, :past_horizon]),
                                        forecasts), dim=1).detach()

visualize_forecasts.plot_time_series(expected_time_series=expected_timeseries[:, 0, 0],
                                     uncertainty=expected_timeseries_outputs[:, :, 0, 0].T)
# Show uncertainty of submodels with heatmap                                     
visualize_forecasts.heatmap_forecasts(expected_timeseries_outputs[:, :, 0, 0].T)
# Show sensitivity for one single output neuron
sensitivity_analysis.sensitivity_analysis(hcnn_ensemble,
                                          Y_batches[:, :past_horizon],
                                          output_neuron=(-1, 10, slice(0, batchsize), 0),
                                          batchsize=2)


# %%
