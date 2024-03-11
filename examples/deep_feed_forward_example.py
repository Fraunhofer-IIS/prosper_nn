# %%
import sys, os

sys.path.append(os.path.abspath(".."))
sys.path.append(os.path.abspath("."))

import torch.optim as optim
import torch.nn as nn
import torch
import prosper_nn.models.deep_feed_forward.deep_feed_forward as deepff
import prosper_nn.utils.sensitivity_analysis as sens_analysis

# %% Parameter
input_dim = 10
output_dim = 1
n_data = 100
deepness = 3
epochs = 100

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %% Create dummy data
X = torch.randn([n_data, input_dim]).to(device)
Y = torch.randn([n_data, output_dim]).to(device)

# %% Initialize Model
deepff = deepff.DeepFeedForward(
    input_dim=input_dim,
    hidden_dim=15,
    output_dim=output_dim,
    deepness=deepness,
    dropout_rate=0.0,
    activation=torch.tanh,
).to(device)

# %% Train Model
optimizer = optim.Adam(deepff.parameters(), lr=0.001)
loss_function = nn.MSELoss()

for epoch in range(epochs):
    epoch_loss = 0
    for x, y in zip(X, Y):
        deepff.zero_grad()

        output = deepff(x)
        loss = loss_function(output, y.repeat(deepness, output_dim))
        epoch_loss += loss
        loss.backward()
        optimizer.step()


# %% Evaluation
# Sensitivity Analysis
sens_analysis.sensitivity_analysis(
    deepff.cpu(), X.cpu(), output_neuron=(0, 0), batchsize=1
)
