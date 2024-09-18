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
import seaborn as sns


def calculate_sensitivity_analysis(
    model: torch.nn.Module,
    *data: Tuple[torch.Tensor, ...],
    output_neuron: tuple = (0,),
    batchsize: int = 1,
) -> torch.Tensor:
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
            while isinstance(y_pred, tuple):
                y_pred = y_pred[output_neuron[0]]
                output_neuron = output_neuron[1:]
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
                # calculate sensitivity for every element of the batch
                for j, y in enumerate(y_pred):
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
            y_pred.backward(retain_graph=False)
            for input_batch in inputs:
                grad = input_batch.grad
                sensitivity_intern = torch.cat([sensitivity_intern, grad], dim=-1)
            sensitivity_intern = sensitivity_intern.unsqueeze(0)
        sensitivity = torch.cat([sensitivity, sensitivity_intern], dim=0)

    return sensitivity


def plot_sensitivity_curve(
    sensitivity: torch.Tensor,
    output_neuron: int = 1,
    xlabel: str = "Observations",
    ylabel: str = "$\\frac{\\partial output}{\\partial input}",
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


# %% Sensitivity for HCNN and other recurrent models
def analyse_temporal_sensitivity(
    model: torch.nn.Module,
    data: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
    task_nodes: List[int],
    n_future_steps: int,
    past_horizon: int,
    n_features: int,
    features: Optional[List[str]] = None,
    xlabel: Optional[str] = "Forecast Step",
    ylabel: Optional[str] = "Features",
    title: Optional[str] = None,
    top_k: int = 1,
    save_at: Optional[str] = None,
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
    task_nodes : List[int]
        The indexes of (target) variables for which the temporal analysis should be done.
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
    top_k : int
        The number of features with the highest absolute/monotonie that are highlighted.

    Returns
    -------
    None
    """
    if features is None:
        features = [f"feature_{i}" for i in range(n_features)]
    if type(data) is torch.Tensor:
        data = (data,)

    # Calculations
    total_heat = []
    for node in task_nodes:
        heat = []
        for time in range(past_horizon, past_horizon + n_future_steps):
            output_neuron = (-1, time, 0, node)

            sensi = calculate_sensitivity_analysis(
                model, *data, output_neuron=output_neuron, batchsize=1
            )
            sensi = sensi.reshape((sensi.shape[0], -1))
            heat.append(sensi[0, (past_horizon - 1) * n_features :])
        heat = torch.stack(heat)

        plot_analyse_temporal_sensitivity(
            heat.T,
            features[node],
            features,
            n_future_steps,
            path=save_at,
            title=title,
            xlabel=xlabel,
            ylabel=ylabel,
            top_k=top_k,
        )
    total_heat.append(heat)

    return torch.stack(total_heat)


def plot_analyse_temporal_sensitivity(
    sensis: torch.Tensor,
    target_var: List[str],
    features: List[str],
    n_future_steps: int,
    path: Optional[str] = None,
    title: Optional[Union[dict, str]] = None,
    xticks: Optional[Union[dict, str]] = None,
    yticks: Optional[Union[dict, str]] = None,
    xlabel: Optional[Union[dict, str]] = None,
    ylabel: Optional[Union[dict, str]] = None,
    figsize: List[float] = [12.4, 5.8],
) -> None:
    """
    Plots a sensitivity analysis and creates a table with monotonie and total heat on the right side
    for each task variable.
    """
    # Calculate total heat and monotony
    total_heat = torch.sum(torch.abs(sensis), dim=2)
    total_heat = (total_heat * 100).round() / 100
    monotonie = torch.sum(sensis, dim=2) / total_heat
    monotonie = (monotonie * 100).round() / 100

    plt.rcParams["figure.figsize"] = figsize
    ### Temporal Sensitivity Heatmap ###
    # plot a sensitivity matrix for every feature/target variable to be investigated
    for i, node in enumerate(target_var):
        # Set description
        if not title:
            title = "Influence of auxiliary variables on {}"
        if not xlabel:
            xlabel = "Weeks into future"
        if not ylabel:
            ylabel = "Auxiliary variables"
        if not xticks:
            xticks = {
                "ticks": range(1, n_future_steps + 1),
                "labels": [
                    str(i) if i % 2 == 1 else None for i in range(1, n_future_steps + 1)
                ],
                "horizontalalignment": "right",
            }
        if not yticks:
            yticks = {
                "ticks": range(len(features)),
                "labels": [feature.replace("_", " ") for feature in features],
                "rotation": 0,
                "va": "top",
                "size": "large",
            }

        sns.heatmap(sensis[i],
                    center=0,
                    cmap='coolwarm',
                    robust=True,
                    cbar_kws={'location':'right', 'pad': 0.22},
                    )
        plt.ylabel(ylabel)
        plt.xlabel(xlabel)
        plt.xticks(**xticks)
        plt.yticks(**yticks),
        plt.title(title.format(node.replace("_", " ")), pad=25)

        # Fade out row name if total heat is not that strong
        for j, ticklabel in enumerate(plt.gca().get_yticklabels()):
            if j >= len(target_var):
                alpha = float(0.5 + (total_heat[i][j] / torch.max(total_heat)) / 2)
                ticklabel.set_color(color=[0, 0, 0, alpha])
            else:
                ticklabel.set_color(color="C0")
        plt.tight_layout()

        ### Table with total heat and monotonie ###
        table_values = torch.stack((total_heat[i], monotonie[i])).T

        # Colour of cells
        cell_colours = [
            ["#E1E3E3" for _ in range(table_values.shape[1])]
            for _ in range(table_values.shape[0])
        ]
        cell_colours[torch.argmax(table_values, dim=0)[0]][0] = "#179C7D"
        cell_colours[torch.argmax(torch.abs(table_values), dim=0)[1]][1] = "#179C7D"

        # Plot table
        plt.table(
            table_values.numpy(),
            loc='right',
            colLabels=['Absolute', 'Monotony'],
            colWidths=[0.2,0.2],
            bbox=[1, 0, 0.3, 1.042],                #[1, 0, 0.4, 1.042],
            cellColours=cell_colours,
            edges='BRT',
            )
        plt.subplots_adjust(left=0.05, right=1.0) # creates space for table

        # Save and close
        if path:
            plt.savefig(
                path + "sensi_analysis_{}.png".format(node), bbox_inches="tight"
            )
        else:
            plt.show()
        plt.close()


# %% Sensitivity for feed-forward models and other not-recurrent models


def sensitivity_analysis(
    model: torch.nn.Module,
    data: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
    output_neuron: Tuple[int, ...],
    batchsize: int,
    xlabel: str = "Observations",
    ylabel: str = "Input Node",
    title: str = "Sensitivity-analysis heatmap for one output neuron",
    cbar_kws: dict = {"label": "d output / d input"},
    save_at: Optional[str] = None,
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
    save_at : str
        Where to save the figure

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
    sensitivity = sensitivity.reshape((sensitivity.shape[0], -1))
    visualization.plot_heatmap(
        sensitivity.T,
        center=0,
        vmin=-torch.max(abs(sensitivity)),
        vmax=torch.max(abs(sensitivity)),
        xlabel=xlabel,
        ylabel=ylabel,
        title=title,
        cbar_kws=cbar_kws,
        save_at=save_at,
        grid={"visible": False},
    )
    return sensitivity
