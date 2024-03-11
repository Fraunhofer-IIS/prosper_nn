import sys, os

sys.path.append(os.path.abspath("."))
import torch
from prosper_nn.models.ensemble import Ensemble
from prosper_nn.models.feedforward import FFNN

# %% Set parameters
batchsize = 10
input_dim = 100
hidden_dim = 20
output_dim = 5
n_models = 5
batchsize = 1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %% Create dummy data
x = torch.randn(input_dim, input_dim).to(device)
y = torch.randn(input_dim, output_dim).to(device)

# %% Initialize Feedforward model and an ensemble of it
ffnn = FFNN(input_dim, hidden_dim, output_dim)
ffnn_ensemble = Ensemble(model=ffnn, n_models=n_models).to(device)

# %% Train model
loss_function = torch.nn.MSELoss()
optimizer = torch.optim.Adam(ffnn_ensemble.parameters(), lr=0.01)

for t in range(500):
    ffnn_ensemble.zero_grad()
    y_preds, mean_y = torch.split(ffnn_ensemble(x), n_models)
    mean_y = mean_y.squeeze(0)

    loss = loss_function(mean_y, y)
    loss.backward()
    optimizer.step()

    if t % 100 == 99:
        print(t, loss_function(mean_y, y).detach())
