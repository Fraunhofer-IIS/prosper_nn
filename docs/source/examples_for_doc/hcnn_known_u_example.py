import prosper_nn.utils.generate_time_series_data as gtsd
import prosper_nn.utils.create_input_ecnn_hcnn as ci
from prosper_nn.models.hcnn_known_u import hcnn_known_u
import torch

# Define network parameters
n_features_U = 10  # setting this to zero reverts to vanilla HCNN with tf
batchsize = 5
past_horizon = 15
forecast_horizon = 5
n_state_neurons = 20
n_data = 50
n_features_Y = 5
sparsity = 0
teacher_forcing = 1
decrease_teacher_forcing = 0.0001

#  Generate data
Y, U = gtsd.sample_data(n_data, n_features_Y, n_features_U)
Y_batches, U_batches = ci.create_input(
    Y,
    past_horizon,
    batchsize,
    U,
    True,  # Has to be true for Hcnn_known_U
    forecast_horizon,
)

Y_batches.shape, U_batches.shape

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

# setting the optimizer, loss and targets
optimizer = torch.optim.Adam(hcnn_known_u_model.parameters(), lr=0.01)
loss_function = torch.nn.MSELoss()
targets = torch.zeros((past_horizon, batchsize, n_features_Y))

# Train model
epochs = 150
for epoch in range(epochs):
    for batch_index in range(0, U_batches.shape[0]):
        hcnn_known_u_model.zero_grad()
        U_batch = U_batches[batch_index]
        Y_batch = Y_batches[batch_index]
        model_out = hcnn_known_u_model(U_batch, Y_batch)
        past_error, forecast = torch.split(model_out, past_horizon)
        loss = loss_function(past_error, targets)
        loss.backward()
        optimizer.step()
