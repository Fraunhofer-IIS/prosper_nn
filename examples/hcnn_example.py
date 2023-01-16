# %% Package Imports
import sys, os
sys.path.append(os.path.abspath('..'))
sys.path.append(os.path.abspath('.'))

import torch
import torch.nn as nn
import torch.optim as optim

import prosper_nn.models.hcnn
from prosper_nn.utils import sensitivity_analysis
from prosper_nn.utils import visualize_forecasts
import prosper_nn.utils.neuron_correlation_hidden_layers as nchl
import prosper_nn.utils.generate_time_series_data as gtsd
import prosper_nn.utils.create_input_ecnn_hcnn as ci

# %% Define network parameters
past_horizon = 10
forecast_horizon = 5
n_features_Y = 2
n_data = 100
n_state_neurons = 3
batchsize = 1
n_batches = 2
sparsity = 0
teacher_forcing = 1
decrease_teacher_forcing = 0.0001

# %% Create data and targets
targets = torch.zeros((past_horizon, batchsize, n_features_Y))

# generate data with "unknown" variables U
Y, U = gtsd.sample_data(n_data, n_features_Y=n_features_Y-1, n_features_U=1)
Y = torch.cat((Y, U), 1)

# Only use Y as input for the hcnn
Y_batches = ci.create_input(Y, past_horizon, batchsize)

# %% Initialize Hcnn
hcnn_model = prosper_nn.models.hcnn.HCNN(
    n_state_neurons,
    n_features_Y,
    past_horizon,
    forecast_horizon,
    sparsity,
    lstm=True, 
    teacher_forcing=teacher_forcing,
    decrease_teacher_forcing=decrease_teacher_forcing)

# %% Train model
optimizer = optim.Adam(hcnn_model.parameters(), lr=0.001)
loss_function = nn.MSELoss()

epochs = 10

total_loss = epochs * [0]
for epoch in range(epochs):
    for batch_index in range(0, Y_batches.shape[0]):
        hcnn_model.zero_grad()

        Y_batch = Y_batches[batch_index]
        model_output = hcnn_model(Y_batch)
        past_error, forecast = torch.split(model_output, past_horizon)

        losses = [loss_function(past_error[i], targets[i]) for i in range(past_horizon)]
        loss = sum(losses)
        loss.backward(retain_graph=True)
        optimizer.step()
        total_loss[epoch] += loss.detach()

# %% Evaluation
# Visualization of the expected timeseries
expected_timeseries = torch.cat((torch.add(past_error.squeeze(), Y[:past_horizon]),
                                forecast.squeeze()), dim=0).detach()

visualize_forecasts.plot_time_series(expected_time_series=expected_timeseries[:, 0],
                                     target=Y[:past_horizon+forecast_horizon, 0])
# Sensitivity Analysis
sensitivity = sensitivity_analysis.sensitivity_analysis(hcnn_model,
                                                        Y_batches,
                                                        output_neuron=(10, slice(0, batchsize), 0),
                                                        batchsize=batchsize)

# Restrict the sensitivity analysis only on one feature
node_for_sensitivity = 0
restricted_sensitivity_matrix = sensitivity[:, range(node_for_sensitivity, past_horizon*n_features_Y, n_features_Y)]
prosper_nn.utils.visualization.plot_heatmap(restricted_sensitivity_matrix.T,
                                 center=0,
                                 vmin=-torch.max(abs(sensitivity)),
                                 vmax=torch.max(abs(sensitivity)),
                                 xlabel={'xlabel': 'Observations'},
                                 ylabel={'ylabel': 'Input Node'},
                                 title={'label': 'Temporal Sensitivity of one Input Node for a Output Neuron'},
                                 cbar_kws={'label': 'd output / d input'},
                                 )

# %% Prediction with trained model
with torch.no_grad():
    hcnn_model.eval()
    output_forecast = hcnn_model(Y_batches[0, :, 0, :].unsqueeze(1))
    forecast = output_forecast[:past_horizon]
    print("Forecast: ".format(forecast))

    # neuron correlation analysis to check size of hidden layer (= n_state_neurons)
    states_for_correlation = torch.empty((Y_batches.shape[0], batchsize, n_state_neurons))
    for batch_index in range(0, Y_batches.shape[0]):
        Y_batch = Y_batches[batch_index]
        model_output = hcnn_model(Y_batch)
        states_for_correlation[batch_index, :, :] = hcnn_model.state[past_horizon]
    states_for_correlation = states_for_correlation.reshape((-1, n_state_neurons))
    corr_matrix, max_corr, ind_neurons = nchl.hl_size_analysis(states_for_correlation)

# %%

# %%
