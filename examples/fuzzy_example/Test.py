import sys, os

sys.path.append(os.path.abspath("."))
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn

from examples.fuzzy_example.dummy_data import createModelData_T
from prosper_nn.models.fuzzy.membership_functions import (
    GaussianMembership,
    NormlogMembership,
)
from prosper_nn.models.fuzzy.fuzzification import Fuzzification
from prosper_nn.models.fuzzy.fuzzy_inference import FuzzyInference
from prosper_nn.models.fuzzy.defuzzification import Defuzzification
from prosper_nn.models.fuzzy.rule_manager import RuleManager


def heatmap_weight_plotter(
    weights: np.array, title: str = None, show_values=False
) -> None:
    """
    Plots a given 2D weight array as a heatmap to show strong and weak connections between two network layers.

    Parameters
    ----------
    weights : np.array
        2D array containing the weights of a layer.
    title : str
        Title to be shown above the figure
    show_values : bool
        If True write array values inside the heatmap pixels.

    Returns
    -------
    None
    """
    heatmap = sns.heatmap(
        np.array(weights).T,
        center=0,
        xticklabels=True,
        cmap="coolwarm",
        robust=True,
        cbar_kws={"label": "weights"},
        annot=show_values,
    )

    heatmap.set_yticklabels(heatmap.get_yticklabels(), rotation=0)
    plt.ylabel("Input Node")
    plt.xlabel("Output Node")
    if title is not None:
        plt.title(title)
    plt.show()


def test_layer(model, data, epochs=50, single=True):
    optimizer = torch.optim.RMSprop(model.parameters(), lr=1e-2)
    criterion = torch.nn.MSELoss(reduction="sum")
    model.train()
    inputs, outputs = data
    losses = []
    for t in range(epochs):
        preds = model(inputs)
        optimizer.zero_grad()
        loss = criterion(preds, outputs)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        if t % 10 == 0:
            print(t, loss.item())

    # Collect the history of W-values and b-values to plot later
    model.train(False)
    plt.plot(range(epochs), losses)
    plt.show()
    if single:
        plt.scatter(np.array(inputs), np.array(outputs), c="b")
        # plt.scatter(inputs, model(inputs), c='r')
        plt.scatter(np.array(inputs), preds.detach().numpy(), c="g")
        plt.show()
    else:
        for i in range(4):
            y_axes = np.array((torch.stack([inputs, inputs, inputs], axis=-1)))
            y = np.array([y_axes[:, i], y_axes[:, i], y_axes[:, i]]).transpose()
            y = np.array(y_axes)[:, i].transpose()
            x_pred = torch.reshape(model(inputs), shape=(1000, 4, 3))
            xp = x_pred.detach().numpy()[:, i].transpose()
            x_out = torch.reshape(outputs, shape=(1000, 4, 3))
            xo = np.array(x_out)[:, i].transpose()
            plt.subplot(2, 2, i + 1)
            plt.scatter(y, xo, c="b")
            plt.scatter(y, xp, c="r")
            # plt.scatter(np.array([inputs, inputs, inputs]).transpose(), preds, c='g')
        plt.show()


def test_model(model, data, epochs=50):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    criterion = torch.nn.CrossEntropyLoss(reduction="mean")
    model.train()
    inputs, labels, lower, upper = data
    losses = []
    optimizer.zero_grad()
    for t in range(epochs):
        preds = model(inputs)
        optimizer.zero_grad()
        loss = criterion(preds, labels)
        loss.backward()
        losses.append(loss.item())
        optimizer.step()

        if t % 100 == 0:
            print(t, loss.item())
    plt.plot(range(epochs), losses)
    plt.show()
    model = model[0]
    for i in range(4):
        y_axes = np.array((torch.stack([inputs, inputs, inputs], axis=-1)))
        y = np.array([y_axes[:, i], y_axes[:, i], y_axes[:, i]]).transpose()
        y = np.array(y_axes)[:, i].transpose()
        x_pred = torch.reshape(model(inputs), shape=(1000, 4, 3))
        xp = x_pred.detach().numpy()[:, i].transpose()
        # x_out = torch.reshape(labels, shape=(1000, 4, 3))
        # xo = np.array(x_out)[:, i].transpose()
        plt.subplot(2, 2, i + 1)
        plt.vlines([lower[i], upper[i]], 0, 1)
        plt.scatter(y, xp, c="r")
        # plt.scatter(np.array([inputs, inputs, inputs]).transpose(), preds, c='g')

    plt.show()


# test_layer(NormlogMember(negative=False), create_normlog_data(1000, 1.5), epochs=100)
membership_fcts = {
    "neg_func": NormlogMembership(negative=True),
    "const_func": GaussianMembership(),
    "pos_func": NormlogMembership(),
}

# test_layer(Fuzzification(n_inputs=4, membership_fcts=membership_fcts), createMemberData(), single=False)
loader = RuleManager(
    "examples/fuzzy_example/Rules.json",
    rule_matrix_shape=(5, 4, 3),
    classification_matrix_shape=(5, 5),
)

model = nn.Sequential(
    Fuzzification(n_features_input=4, membership_fcts=membership_fcts),
    FuzzyInference(
        n_features_input=4,
        n_rules=5,
        n_membership_fcts=3,
        n_output_classes=5,
        rule_matrix=loader.rule_matrix,
        activation=torch.nn.Tanh(),
        prune_weights=True,
        learn_conditions=False,
        classification_matrix=loader.classification_matrix,
    ),
    Defuzzification(
        n_output_classes=5,
        n_features_output=5,
    ),
)
model.double()

# Show conditions and consequences before training
conditions = np.reshape(model[1].conditions.weight.detach().numpy(), (5, 4 * 3))
consequences = model[1].consequences.weight.detach().numpy()
heatmap_weight_plotter(consequences, "Rules to Error Classes", show_values=True)
heatmap_weight_plotter(conditions, title="Rule Weights", show_values=True)

# Train model and show conditions and consequences again
test_model(model, createModelData_T(loader.rule_matrix, 1000, 4, 5), epochs=1500)
conditions = np.reshape(model[1].conditions.weight.detach().numpy(), (5, 4 * 3))
consequences = model[1].consequences.weight.detach().numpy()
heatmap_weight_plotter(conditions, title="Rule Weights", show_values=True)
heatmap_weight_plotter(consequences, "Rules to Error Classes", show_values=True)
