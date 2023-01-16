import sys, os
sys.path.append(os.path.abspath('.'))

import torch
import torch.nn as nn
import torch.optim as optim
from collections import OrderedDict
import prosper_nn.utils.neuron_correlation_hidden_layers as nchl

# %% Create dummy data
n_data = 10
X = torch.rand([n_data, 100])
Y = torch.rand([n_data, 1])

# %% Initialize model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(100, 5)
        self.act1 = nn.Tanh()
        self.fc2 = nn.Sequential(OrderedDict([
            ('fc2_1', nn.Linear(5, 20)),
            ('fc2_2', nn.Tanh()),
            ('fc2_3', nn.Linear(20, 10)),
            ('fc2_4', nn.Tanh())
            ]))
        self.fc3 = nn.Linear(10, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x

net = Net()

# %% Train model
optimizer = optim.Adam(net.parameters(), lr=0.001)
loss_function = nn.MSELoss()

# Learning of the parameters
epochs = 10
for epoch in range(epochs):
    for x, y in zip(X, Y):
        net.zero_grad()

        output = net(x)

        loss = loss_function(output, y)
        loss.backward()
        optimizer.step()

# %% Evaluation
# Hidden layer correlation
with torch.no_grad():
    net.eval()

    # Variant where the layer output is manually entered
    x = net.fc1(X)
    x = net.act1(x)
    output_layer = net.fc2(x)
    corr_matrix1, max_corr1, ind_neurons1 = nchl.hl_size_analysis(output_layer)

    # Variant where the index or name of the layer is forwarded
    corr_matrix2, max_corr2, ind_neurons2 = nchl.hl_size_analysis_Sequential(net, 
                                                                             name_layer="act1", 
                                                                             model_input=X)
