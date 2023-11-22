# %%
import sys, os

sys.path.append(os.path.abspath(".."))
sys.path.append(os.path.abspath("."))

import torch
import torch.nn as nn
import torch.optim as optim
import prosper_nn.models.ensemble as ensemble
import prosper_nn.utils.generate_time_series_data as gtsd
import prosper_nn.utils.create_input_ecnn_hcnn as ci
from prosper_nn.utils import visualize_forecasts
from prosper_nn.models.hcnn_known_u import hcnn_known_u

# %%
# Define network parameters
n_features_U = 2
batchsize = 5
past_horizon = 15
forecast_horizon = 4
future_U = True  # Has to be true for Hcnn_known_U
n_state_neurons = 30
n_data = 50
n_features_Y = 3
sparsity = 0
teacher_forcing = 1
decrease_teacher_forcing = 0.0001
n_models = 10

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %%
#  Generate data
Y, U = gtsd.sample_data(n_data, n_features_Y, n_features_U)
Y_batches, U_batches = ci.create_input(
    Y, past_horizon, batchsize, U, future_U, forecast_horizon
)

Y_batches = Y_batches.to(device)
U_batches = U_batches.to(device)


# %%
# Initialize HCNN_KNOWN_U
hcnn_known_u_model = hcnn_known_u.HCNN_KNOWN_U(
    n_state_neurons,
    n_features_U,
    n_features_Y,
    past_horizon,
    forecast_horizon,
    sparsity,
    teacher_forcing=teacher_forcing,
    decrease_teacher_forcing=decrease_teacher_forcing,
)

hcnn_known_u_ensemble = ensemble.Ensemble(
    model=hcnn_known_u_model,
    n_models=n_models,
    sparsity=sparsity,
    keep_pruning_mask=False,
    initializer=torch.nn.init.kaiming_uniform_,
).to(device)

# %%
# setting the optimizer, loss and targets
optimizer = optim.Adam(hcnn_known_u_ensemble.parameters(), lr=0.01)
loss_function = nn.MSELoss()
targets = torch.zeros((past_horizon, batchsize, n_features_Y), device=device)
# %%
# Train model
epochs = 10
total_loss = epochs * [0]
for epoch in range(epochs):
    for batch_index in range(0, U_batches.shape[0]):
        hcnn_known_u_ensemble.zero_grad()

        U_batch = U_batches[batch_index]
        Y_batch = Y_batches[batch_index]

        ensemble_output = hcnn_known_u_ensemble(U_batch, Y_batch)

        outputs, mean = torch.split(ensemble_output, n_models)

        mean = torch.squeeze(mean, 0)
        past_errors, forecasts = torch.split(outputs, past_horizon, dim=1)
        loss = loss_function(mean[:past_horizon], targets)

        loss.backward()
        optimizer.step()
        total_loss[epoch] += loss.detach()

# %% Create Forecast
U_forecast = U_batches[0, :, 0].unsqueeze(1)
Y_forecast = Y_batches[0, :, 0].unsqueeze(1)

with torch.no_grad():
    hcnn_known_u_ensemble.eval()

    ensemble_output = hcnn_known_u_ensemble(U_forecast, Y_forecast)
    outputs, mean = torch.split(ensemble_output, n_models)
    past_errors, forecasts = torch.split(outputs, past_horizon, dim=1)
    mean_past_error, mean_forecast = torch.split(mean, past_horizon, dim=1)

    print("Forecast: \n{}".format(mean_forecast))


# %%
# work in progress--
# Visualization of the expected timeseries
expected_timeseries = (
    torch.cat(
        (
            torch.add(mean_past_error.squeeze(0)[:past_horizon], Y_forecast),
            mean_forecast.squeeze(0),
        ),
        dim=0,
    )
    .squeeze(1)
    .cpu()
)
expected_timeseries_outputs = (
    torch.cat((torch.add(past_errors, Y_forecast), forecasts), dim=1).squeeze(2).cpu()
)

# plot for first feature of Y
visualize_forecasts.plot_time_series(
    expected_time_series=expected_timeseries[:, 0],
    target=Y_forecast.squeeze(1)[:, 0].cpu(),
    uncertainty=expected_timeseries_outputs[:, :, 0].T,
)
# Show uncertainty of submodels with heatmap
visualize_forecasts.heatmap_forecasts(forecasts=expected_timeseries_outputs[:, :, 0].T)
