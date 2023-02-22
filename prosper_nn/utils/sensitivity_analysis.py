""""""
"""
Prosper_nn provides implementations for specialized time series forecasting
neural networks and related utility functions.

Copyright (C) 2022 Nico Beck, Julia Schemm, Henning Frechen, Jacob Fidorra,
    Denni Schmidt, Sai Kiran Srivatsav Gollapalli

This file is part of Propser_nn.

Propser_nn is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

import torch
import matplotlib.pyplot as plt
from . import visualization
from typing import Optional, List, Tuple, Union


def calculate_sensitivity_analysis(
    model: torch.nn.Module,
    *data: Tuple[torch.Tensor, ...],
    output_neuron: tuple = (0,),
    batchsize: int = 1
) -> torch.tensor:
    """
    Calculates the sensitivity matrix.
    The function differentiates the target node with respect to the
    input for all observation.

    Parameters
    ----------
    model : torch.nn.Module
        The PyTorch model for whose output the sensitivity analysis is done.
    data : tuple of PyTorch tensors
        The data input for the model for which the sensitivity analysis
        is done. Should be a tuple, even if it only has one element.
    output_neuron : tuple
        Choose the output node for which the sensitivity analysis should be performed.
        The tuple is used to navigate in the model output to the wished output node.
        For example a tuple (0, 1, 3) is applied on the model output in the following way:
        wished_output_neuron = model_output[0][1][3].
        If there is a batch dimension in the data, insert "slice(0, batchsize)" in the
        corresponding position of the tuple.
        Otherwise the values should be natural non-negative numbers.
    batchsize:
        The batchsize of the model and the data.

    Returns
    -------
    torch.tensor
        A torch tensor with the value of the model output differentiated with
        respect to the model input, evaluated for all observations in data.
        So the output shape of the returned torch tensor is dependent on data.shape()
        (if the model only takes one input, these shapes are equal).
    """

    sensitivity = torch.empty((0,))
    n_batches = data[0].shape[0]
    for i in range(n_batches):  # calculate sensitivity for every batch
        sensitivity_intern = torch.empty((0,))
        inputs = []
        # pass batches through model
        for input in data:
            input_batch = input[i]
            input_batch.requires_grad_(True)
            inputs.append(input_batch)
        y_pred = model(*inputs)
        # get output for output neuron
        try:
            y_pred = y_pred[output_neuron]
        except AttributeError:
            raise AttributeError("Output neuron could not be found in model output.")
        for input_batch in inputs:
            if input_batch.grad is not None:
                input_batch.grad.zero_()
        if batchsize > 1:
            # get grad
            for input_batch in inputs:
                sensitivity_intern_batch = torch.empty((0,))
                for j, y in enumerate(
                    y_pred
                ):  # calculate sensitivity for every element of the batch
                    y.backward(retain_graph=True)
                    grad = input_batch.grad[j]
                    grad = grad.unsqueeze(0)
                    sensitivity_intern_batch = torch.cat(
                        [sensitivity_intern_batch, grad], dim=0
                    )
                sensitivity_intern = torch.cat(
                    [sensitivity_intern, sensitivity_intern_batch], dim=-1
                )
        else:
            # get grad
            y_pred.backward(retain_graph=True)
            for input_batch in inputs:
                grad = input_batch.grad
                sensitivity_intern = torch.cat([sensitivity_intern, grad], dim=-1)
            sensitivity_intern = sensitivity_intern.unsqueeze(0)
        sensitivity = torch.cat([sensitivity, sensitivity_intern], dim=0)
    sensitivity = sensitivity.reshape((sensitivity.shape[0], -1))

    return sensitivity


def plot_sensitivity_curve(
    sensitivity: torch.tensor,
    output_neuron: int = 1,
    xlabel: str = "Observations",
    ylabel: str = "d output / d input",
    title: str = "Sensitivity analysis of one output node",
) -> None:
    """
    A plotting function for a two dimensional matrix.
    It can be used to plot the sensitivity for one individual output node as a graph.

    Parameters
    ----------
    sensitivity : torch.Tensor
        In general a two dimensional torch tensor with float values.
        Here in particular the sensitivity matrix for a neural network
        and the corresponding observations.
    output_neuron : int
        The index in the output layer for which the sensitivity should be plotted.
    xlabel : dict
        Set the label for the x-axis.
    ylabel : dict
        Set the label for the y-axis.
    title : dict
        Set a title for the axes.

    Returns
    -------
    None
    """

    node_sensitivity = sensitivity.T[output_neuron]
    # generate figure
    _, ax = plt.subplots()
    ax.plot(range(len(node_sensitivity)), node_sensitivity)
    ax.set(
        xlabel=xlabel,
        ylabel=ylabel,
        title=title,
        xticks=[],
    )
    plt.tick_params(left="off")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid()

    plt.show()


def analyse_temporal_sensitivity(
    model: torch.nn.Module,
    data: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
    n_task_nodes: int,
    n_future_steps: int,
    past_horizon: int,
    n_features: int,
    features: Optional[List[str]] = None,
    xlabel: Optional[str] = "Features",
    ylabel: Optional[str] = "Time",
    title: Optional[str] = None,
) -> None:
    """
    Function to analyze the influence of the present on the future variables in HCNNs and other
    recurrent type models. For each task variable the sensitivity of all features is plotted for
    different times in the future. It is necessary that the batchsize of the data is 1.

    Parameters
    ----------
    model : torch.Module
        The PyTorch model for whose output the sensitivity analysis is done.
        The model should be an ensemble with
        output shape = (n_models, past_horizon + forecast_horizon, batchsize=1, n_features_Y).
    data : torch.Tensor or tuple of torch.Tensors
        The data input for the model for which the sensitivity analysis
        is performed. E.g., for an ensemble of HCNNs this might be a tensor and for an ensemble of
        ECNNs this might be a tuple of tensors (in the right order to give to the ECNN).
        Each input tensor for the model is assumed to be of the
        shape=(n_batches, past_horizon, batchsize, n_features).
        If your model needs input of a different shape, you might have to adapt the code.
    n_task_nodes : int
        The amount of (target) variables for which the temporal analysis should be done.
        The first n_task_nodes features are taken.
    n_future_steps : int
        The number of forecasting steps that are investigated by the analysis.
    past_horizon : int
        The past horizon gives the number of time steps into the past that are used for forecasting.
    n_features : int
        The size of the data/number of features in each time step.
    features : list[str], optional
        The names of the features in the data.
    xlabel : str, optional
        Set the label for the x-axis.
    ylabel : str, optional
        Set the label for the y-axis.
    title : list[str], optional
        Set a title for the plot.

    Returns
    -------
    None
    """
    if type(data) is torch.Tensor:
        data = (data,)
    # Calculations
    heat = torch.empty((n_task_nodes, n_future_steps, n_features))
    for node in range(0, n_task_nodes):
        for i, time in enumerate(
            range(past_horizon + 1, past_horizon + 1 + n_future_steps)
        ):
            sensi = calculate_sensitivity_analysis(
                model, *data, output_neuron=(-1, time, 0, node), batchsize=1
            )
            heat[node][i] = sensi[0, (past_horizon - 1) * n_features :]

    # Visualization
    if features is None:
        features = [str(i) for i in range(n_features)]
    if title is None:
        title = [
            "Influence of present features on future output node {}.".format(
                features[node]
            )
            for node in range(n_task_nodes)
        ]
    yticks = {
        "ticks": range(1, n_future_steps + 1),
        "labels": ["+" + str(i) for i in range(1, n_future_steps + 1)],
        "rotation": 90,
    }
    xticks = {
        "ticks": range(n_features),
        "labels": features,
        "horizontalalignment": "left",
    }
    # plot a sensitivity matrix for every feature/target variable to be investigated
    for node in range(0, n_task_nodes):
        visualization.plot_heatmap(
            heat[node],
            center=0,
            xlabel=xlabel,
            ylabel=ylabel,
            title=title[node],
            xticks=xticks,
            yticks=yticks,
        )
    return


def sensitivity_analysis(
    model: torch.nn.Module,
    data: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
    output_neuron: Tuple[int, ...],
    batchsize: int,
    xlabel: str = "Observations",
    ylabel: str = "Input Node",
    title: str = "Sensitivity-analysis heatmap for one output neuron",
    cbar_kws: dict = {"label": "d output / d input"},
) -> torch.Tensor:
    """
    Sensitivity for feed-forward models and other not-recurrent models.
    The function differentiates the target node with respect to the
    input for all observation.
    In this way the influence of each input neuron on the selected output neuron can
    be investigated.
    Combines the calculation of the sensitivity matrix and the
    visualization in a heatmap in one function.

    Parameters
    ----------
    model : torch.nn.Module
        The PyTorch model for whose output the sensitivity analysis is done.
    data : torch.Tensor or tuple of torch.Tensors
        The data input for the model for which the sensitivity analysis
        is performed. Depending on how many inputs the model takes for the forward pass,
        data is either a tensor or a tuple of tensors.
        It's assumed that every input tensor for the model has the
        shape=(n_batches, batchsize, n_features).
        For MLPs you can also use data in the shape of (n_observations, batchsize, n_features).
        If your model needs input of a different shape, you might have to adapt the code.
    output_neuron : tuple
        Choose the output node for which the sensitivity analysis should be performed.
        The tuple is used to navigate in the model output to the wished output node.
        For example a tuple (0, 1, 3) is applied on the model output in the following way:
        wished_output_neuron = model_output[0][1][3].
        If there is a batch dimension in the data, insert "slice(0, batchsize)" in the
        corresponding position of the tuple.
        All the other values should be natural non-negative numbers.
    batchsize: int
        The batchsize of the model and the data.
    xlabel : str
        Set the label for the x-axis.
    ylabel : str
        Set the label for the y-axis.
    title : str
        Set a title for the axes.
    cbar_kws : dict
        Keyword arguments for matplotlib.figure.Figure.colorbar().

    Returns
    -------
    torch.Tensor
        A torch tensor with the value of the model output differentiated with
        respect to the model input, evaluated for all observations in data.
        The output shape of the returned torch tensor is dependent on data.shape()
        (if the model only takes one input, these shapes are equal).
    """

    if type(data) is torch.Tensor:
        data = (data,)
    sensitivity = calculate_sensitivity_analysis(
        model, *data, output_neuron=output_neuron, batchsize=batchsize
    )
    visualization.plot_heatmap(
        sensitivity.T,
        center=0,
        vmin=-torch.max(abs(sensitivity)),
        vmax=torch.max(abs(sensitivity)),
        xlabel=xlabel,
        ylabel=ylabel,
        title=title,
        cbar_kws=cbar_kws,
    )
    return sensitivity
