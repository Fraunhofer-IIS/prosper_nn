import torch
from prosper_nn.models.deep_feed_forward.deep_feed_forward import DeepFeedForward

# Set model and data parameter
input_dim = 10
hidden_dim = 15
output_dim = 1
n_batches = 100
batchsize = 5
deepness = 3

# Initialise Deep Feedforward Neural Network
deepff = DeepFeedForward(input_dim=input_dim,
                         hidden_dim=hidden_dim,
                         output_dim=output_dim,
                         deepness=deepness)

X = torch.randn([n_batches, batchsize, input_dim])
Y = torch.randn([n_batches, batchsize, output_dim])

# Train Model
optimizer = torch.optim.Adam(deepff.parameters())
loss_function = torch.nn.MSELoss()

for epoch in range(10):
    for x, y in zip(X, Y):
        output = deepff(x)

        deepff.zero_grad()
        loss = sum([loss_function(output[i], y) for i in range(deepness)]) / deepness
        loss.backward()
        optimizer.step()