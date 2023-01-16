import sys, os
sys.path.append(os.path.abspath('.'))
import torch
from prosper_nn.models.ensemble import Ensemble
from prosper_nn.models.feedforward import FFNN
from prosper_nn.utils import sensitivity_analysis

# %% Set parameters
batchsize = 10
input_dim = 100
hidden_dim = 20
output_dim = 5
n_models = 5
batchsize = 1

# %% Create dummy data
x = torch.randn(input_dim, input_dim)
y = torch.randn(input_dim, output_dim)

# %% Initialize Feedforward model and an ensemble of it
ffnn = FFNN(input_dim, hidden_dim, output_dim)
ffnn_ensemble = Ensemble(model=ffnn, n_models=n_models)

# %% Train model
loss_function = torch.nn.MSELoss()
optimizer = torch.optim.Adam(ffnn_ensemble.parameters(), lr=0.01)

for t in range(500):
    ffnn_ensemble.zero_grad()
    y_preds, mean_y = torch.split(ffnn_ensemble(x), n_models)
    
    loss = sum([loss_function(y_pred, y) for y_pred in y_preds]) / n_models
    loss.backward()
    optimizer.step()

    if t % 100 == 99:
        print(t, loss_function(mean_y, y).detach())
