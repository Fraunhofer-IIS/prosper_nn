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
import torch.nn as nn
import numpy as np
from typing import List, Optional, Union, Tuple
from . import visualization


def corrcoef(m: torch.Tensor, rowvar: bool = True) -> torch.Tensor:
    """Return Pearson product-moment correlation coefficients.

    Implementation of the numpy function of the same name.

    Parameters
    ----------
        m: A 1-D or 2-D torch.Tensor containing multiple variables and observations.
            Each row of `m` represents a variable, and each column a single
            observation of all those variables.
        rowvar: If `rowvar` is True, then each row represents a
            variable, with observations in the columns. Otherwise, the
            relationship is transposed: each column represents a variable,
            while the rows contain observations.

    Returns
    -------
    torch.Tensor
        The correlation matrix of the variables.
    """
    if m.dim() > 2:
        raise ValueError("m has more than 2 dimensions")
    if m.dim() < 2:
        m = m.view(1, -1)
    if not rowvar and m.size(0) != 1:
        m = m.t()

    # m = m.type(torch.double)  # uncomment this line if desired
    fact = 1.0 / (m.size(1) - 1)
    m -= torch.mean(m, dim=1, keepdim=True)
    mt = m.t()
    cov = fact * m.matmul(mt).squeeze()

    try:
        d = torch.diag(cov)
    except ValueError:
        # scalar covariance
        # nan if incorrect value (nan, inf, 0), 1 otherwise
        return cov / cov
    stddev = torch.sqrt(d)
    corr = cov / stddev[:, None]
    corr = corr / stddev[None, :]

    return corr


def _create_corr_matrix(output_layer: torch.Tensor) -> torch.Tensor:
    """
    Creates a correlation matrix of layer neurons with corrcoef function.

    Parameters
    ----------
    output_layer : torch.Tensor
        The output of the layer you want to investigate.
        shape = (n_neurons, sample size).

    Returns
    -------
    corr_matrix : torch.tensor
        A 2D tensor, the correlation matrix of the neurons in the layer.

    """

    # transpose output_layer to iterate on the rows as they represent
    # the neurons in the layer
    output_layer_t = torch.transpose(output_layer, 0, 1)
    corr_matrix = corrcoef(output_layer_t)
    corr_matrix = torch.round(corr_matrix * 100) / 100

    return corr_matrix


def _determine_annotations(
    corr_matrix: torch.tensor, min_absolute_corr: float
) -> np.array:
    """
    Determines and formats annotations for plotting of the correlation matrix.

    Parameters
    ----------
    corr_matrix : torch.Tensor
        The correlation matrix of the neurons in the layer.
    min_absolute_corr : float
        A value between 0 and 1 which defines the minimal absolute correlation
        coefficient that is to be displayed in the plotted correlation matrix.


    Returns
    -------
    np.array
    """
    annotations = corr_matrix.clone()
    annotations[
        (annotations > -min_absolute_corr) * (annotations < min_absolute_corr)
    ] = 0
    annotations = annotations.to("cpu").numpy()
    annotations = np.where(np.equal(annotations, 0), "", annotations)
    return annotations


def _find_max_correlation(
    corr_matrix: torch.tensor, print_values: bool
) -> Union[List[torch.Tensor], Tuple[torch.Tensor]]:
    """
    Finds the strongest correlation and the corresponding neurons in correlation matrix.

    Parameters
    ----------
    corr_matrix : torch.Tensor
        The correlation matrix of the neurons in the layer.
    print_values : boolean
        Are the value of the strongest correlation and the corresponding
        neuron indices to be printed?

    Returns
    -------
    List[torch.Tensor]
       The first entry of the tuple is a tensor which contains the correlation
       coefficient(s) indicating the strongest correlation. The second entry
       is a tensor containing the indices of the corresponding neurons.

    """

    # find absolute max entry NOT on diagonal
    corr_matrix_0diag = torch.tril(corr_matrix.clone())
    corr_matrix_0diag = corr_matrix_0diag.fill_diagonal_(0)
    abs_max_corr = torch.max(torch.abs(corr_matrix_0diag))

    # find indeces of most correlated neurons
    ind_neurons_pos = torch.nonzero(corr_matrix_0diag == abs_max_corr, as_tuple=False)
    ind_neurons_neg = torch.nonzero(corr_matrix_0diag == -abs_max_corr, as_tuple=False)
    ind_neurons = torch.cat((ind_neurons_pos, ind_neurons_neg), 0)

    abs_max_corr = abs_max_corr.tolist()
    abs_max_corr = round(abs_max_corr, 2)

    print_most_corr = "The most correlated neurons are the ones with indices "
    if ind_neurons_neg.nelement() == 0:
        if print_values:
            print(print_most_corr, ind_neurons.squeeze().tolist(),)
            print("The according Pearson correlation coefficient is", abs_max_corr)
        return (abs_max_corr, ind_neurons)
    elif ind_neurons_pos.nelement() == 0:
        if print_values:
            print(
                print_most_corr,
                ind_neurons.squeeze().tolist(),
            )
            print("The according Pearson correlation coefficient is", -abs_max_corr)
        return (-abs_max_corr, ind_neurons)
    else:
        if print_values:
            print(
                print_most_corr,
                ind_neurons.squeeze().tolist(),
            )
            print(
                "The according Pearson correlation coefficients are + -",
                abs_max_corr.tolist(),
            )
        return [torch.tensor([[-abs_max_corr], [abs_max_corr]]), ind_neurons]


def hl_size_analysis(
    output_layer: torch.Tensor,
    min_absolute_corr: float = 0.5,
    print_values: bool = True,
    xlabel: str = "Neuron Index",
    ylabel: str = "Neuron Index",
    title: str = "Correlation of neurons in layer",
) -> List[torch.Tensor]:
    """
    Analyses the correlation of neurons in a layer to see if more neurons are
    needed. If the strongest correlation is small, it can be
    helpful to increase the number of neurons.
    Plots correlation matrix of layer neurons, gives correlation matrix and
    strongest correlation with corresponding neurons.

    Parameters
    ----------
    output_layer : torch.Tensor
        The output of the layer you want to investigate.
    min_absolute_corr : float
        See plot_correlation. The default is 0.5.
    print_values : boolean
        See find_max_correlation.
    xlabel : str
        Set the label for the x-axis.
    ylabel : str
        Set the label for the y-axis.
    title : str
        Set a title for the axes.

    Returns
    -------
    List[torch.Tensor]
        Contains the correlation matrix as a 2D PyTorch tensor, the value of
        the strongest correlation  and the indices of the
        corresponding neurons.
    """
    corr_matrix = _create_corr_matrix(output_layer)

    # plot heatmap
    mask = np.zeros_like(corr_matrix)
    mask[np.triu_indices_from(mask)] = True
    visualization.plot_heatmap(
        corr_matrix,
        xlabel=xlabel,
        ylabel=ylabel,
        title=title,
        cbar_kws={"label": "Pearson Corr Coef"},
        vmin=-1,
        vmax=1,
        annot=_determine_annotations(corr_matrix, min_absolute_corr),
        mask=mask,
        fmt="",
        square=True,
        figsize=(10, 10),
    )

    max_corr, ind_neurons = _find_max_correlation(corr_matrix, print_values)

    return [corr_matrix, max_corr, ind_neurons]


def hl_size_analysis_Sequential(
    loaded_model: nn.Sequential,
    model_input: torch.Tensor,
    index_layer: Optional[int] = None,
    name_layer: Optional[str] = None,
    min_absolute_corr: float = 0.5,
    print_values: bool = True,
    xlabel: str = "Neuron Index",
    ylabel: str = "Neuron Index",
    title: str = "Correlation of neurons in layer",
) -> List[torch.Tensor]:
    """
    Analyses the correlation of neurons in a layer to see if more neurons are
    needed. If the strongest correlation is small, it can be
    helpful to increase the number of neurons.
    Plots correlation matrix of layer neurons, gives correlation matrix and
    strongest correlation with corresponding neurons. In contrast to
    hl_size_analysis, the layer output is automatically computed, using the
    loaded model, the model input and the name OR the index of the
    layer(module) in the model.

    Be careful, always check the print-out to see if you are actually analyzing the model
    you want to analyze. If the layers in your model are not initialized in the order they are used
    in the sequential, you might analyze the wrong model.

    Parameters
    ----------
    loaded_model : nn.Sequential
        The loaded model which you want to analyze.
    model_input : torch.Tensor
        The input for the model.
    index_layer : integer
        The index of the layer you want to analyze in
        list(loaded_model.modules()). Bear in mind that the first entry in
        this list is the pre-trained model itself and that submodules cannot be
        analyzed individually but only the top level module they are part of.
        EITHER index_layer OR name_layer has to be forwarded.
    name_layer : string
        The name of the layer you want to analyze. For this to suffice,
        all modules at the top level have to be named. Bear in mind that
        submodules cannot be analyzed individually but only the top level
        module they are part of.
        EITHER index_layer OR name_layer has to be forwarded.
    min_absolute_corr : float
        See plot_correlation.
    print_values : boolean
        See find_max_correlation.
    xlabel : str
        Set the label for the x-axis.
    ylabel : str
        Set the label for the y-axis.
    title : str
        Set a title for the axes.

    Returns
    -------

    List[torch.Tensor]
        Contains the correlation matrix as a 2D PyTorch tensor, the value of
        the strongest correlation  and the indices of the
        corresponding neurons.

    """

    if index_layer is None:
        if name_layer is None:
            raise ValueError(
                "Error: Either name_layer or index_layer has to be forwarded."
            )
        else:
            index_layer = (
                list(dict(loaded_model.named_modules()).keys()).index(name_layer) + 1
            )
    else:
        if name_layer is not None:
            raise ValueError(
                "Error: EITHER name_layer OR index_layer has to be forwarded."
            )

    # copy loaded model up until the layer that is to be analyzed
    shorter_model = torch.nn.Sequential(*list(loaded_model.modules())[1:index_layer])
    print("The analysis refers to the last module of the following model: ")
    print(shorter_model)
    with torch.no_grad():
        output_shorter_model = shorter_model(model_input)
    corr_matrix = _create_corr_matrix(output_shorter_model)

    mask = np.zeros_like(corr_matrix)
    mask[np.triu_indices_from(mask)] = True
    visualization.plot_heatmap(
        corr_matrix,
        xlabel=xlabel,
        ylabel=ylabel,
        title=title,
        cbar_kws={"label": "Pearson Corr Coef"},
        vmin=-1,
        vmax=1,
        annot=_determine_annotations(corr_matrix, min_absolute_corr),
        mask=mask,
        fmt="",
        square=True,
        figsize=(10, 10),
    )

    max_corr, ind_neurons = _find_max_correlation(corr_matrix, print_values)

    return [corr_matrix, max_corr, ind_neurons]
