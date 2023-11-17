import torch
from prosper_nn.models.feedforward.feedforward import FFNN

# Set model and data parameter
input_dim = 10
hidden_dim = 15
output_dim = 1
n_batches = 100
batchsize = 5

# Initialise Deep Feedforward Neural Network
feedforward = FFNN(input_dim=input_dim,
                   hidden_dim=hidden_dim,
                   output_dim=output_dim)

X = torch.randn([n_batches, batchsize, input_dim])
Y = torch.randn([n_batches, batchsize, output_dim])

# Train Model
optimizer = torch.optim.Adam(feedforward.parameters())
loss_function = torch.nn.MSELoss()

for epoch in range(10):
    for x, y in zip(X, Y):
        output = feedforward(x)

        feedforward.zero_grad()
        loss = loss_function(output, y)
        loss.backward()
        optimizer.step()