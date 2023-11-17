import sys, os

# from models import hcnn
sys.path.append(os.path.abspath(".."))
sys.path.append(os.path.abspath("."))

import torch
import torch.nn as nn
import torch.optim as optim
import prosper_nn.utils.generate_time_series_data as gtsd
import prosper_nn.utils.create_input_ecnn_hcnn as ci
from prosper_nn.models.ecnn.ecnn import ECNN
from prosper_nn.utils import visualize_forecasts
from prosper_nn.models.autoencoder.autoencoder import Autoencoder

import matplotlib.pyplot as plt


# %% Define network parameters


n_target_features = 10  # overall number of outputs / outputs of autoencoder
n_features_U = 5  # input features of ecnn
future_U = True
n_state_neurons = 4  # hidden neurons of ecnn
past_horizon = 10  # past horizon of ecnn
forecast_horizon = 3  # forcast horizon of ecnn
n_features_Y = 3  # output features of ecnn / hidden neurons of autoencoder
batchsize = 5
init_state = torch.zeros(1, n_state_neurons)


# %% Initialise ECNN, autoencoder

# autoencoder
autoencoder = Autoencoder(n_target_features, n_features_Y)


# ecnn
ecnn = ECNN(
    n_features_U=n_features_U,
    n_state_neurons=n_state_neurons,
    past_horizon=past_horizon,
    forecast_horizon=forecast_horizon,
    cell_type="ecnn_cell",
    approach="backward",
    init_state=init_state,
    learn_init_state=True,
    n_features_Y=n_features_Y,
    future_U=future_U,
)

# module list to combine parameters of both models for training
modules = nn.ModuleList([autoencoder, ecnn])

# %% Creating dataset and targets

n_data = 1500
Y, U = gtsd.sample_data(n_data, n_target_features, n_features_U)
Y_batches, U_batches = ci.create_input(
    Y, past_horizon, batchsize, U, future_U, forecast_horizon
)

# targets are 0 because of error correction architecture of ecnn
targets = torch.zeros((past_horizon, batchsize, n_features_Y))

# %% Train model

# optimizer and error function
optimizer = optim.Adam(
    modules.parameters(), lr=0.01
)  # all parameters because of module list
loss_function = nn.MSELoss()
epochs = 10

# loss array
total_loss = epochs * [0]

for epoch in range(epochs):
    for batch_index in range(0, U_batches.shape[0]):
        modules.zero_grad()
        U_batch = U_batches[batch_index]
        Y_batch = Y_batches[batch_index]

        # passing Y through autoencoder
        autoencoder_output = autoencoder(Y_batch)
        # using autoencoder to compress Y
        Y_compressed = autoencoder.encode(Y_batch)

        # ecnn pass through
        ecnn_output = ecnn(U_batch, Y_compressed)
        past_error, forecast = torch.split(ecnn_output, past_horizon)

        # loss for ecnn -> output should be zero / compare to zero target
        losses_ecnn = [
            loss_function(past_error[i], targets[i]) for i in range(past_horizon)
        ]
        # loss for autoencoder -> output should be input
        losses_autoencoder = [
            loss_function(autoencoder_output[i], Y_batch[i])
            for i in range(past_horizon)
        ]
        # overall loss
        loss = sum(losses_ecnn) + sum(losses_autoencoder)
        loss.backward()
        optimizer.step()
        total_loss[epoch] += loss.detach()
    if epoch % 5 == 0:
        print("epoch {}/{} completed. Loss: {}".format(epoch, epochs, loss.detach()))

# plot loss
plt.plot(total_loss)


# %% Example data for prediction test

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
    (past_horizon + forecast_horizon, 1, n_target_features),
).float()


# %% Predict with trained model

with torch.no_grad():
    # preparing model
    ecnn.eval()
    autoencoder.eval()
    ecnn.batchsize = 1

    # using autoencoder to compress Y
    example_pred_compressed_Y = autoencoder.encode(example_pred_Y)

    # feeding compressed Y and input U through the ecnn model
    ecnn_output = ecnn(example_pred_U, example_pred_compressed_Y[0:past_horizon])
    past_predictions_compressed, forecast_compressed = torch.split(
        ecnn_output, past_horizon
    )

    # IMPORTANT FOR FORECAST / DIFFERENT TO TRAINING
    # using autoencoder to decompress ecnn output
    past_predictions = autoencoder.decode(past_predictions_compressed)
    forecast = autoencoder.decode(forecast_compressed)

    # adding Y to model's past prediction and concatenating it with the model's forecast
    forecast_timeseries = (
        torch.cat(
            (torch.add(past_predictions, example_pred_Y[:past_horizon]), forecast),
            dim=0,
        )
        .detach()
        .squeeze()
    )
    # print forecast results
    visualize_forecasts.plot_time_series(
        forecast_timeseries[:, 0], example_pred_Y.squeeze(1)[:, 0]
    )
