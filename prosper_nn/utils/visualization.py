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
import seaborn as sns
import numpy as np
from typing import Optional


def plot_heatmap(
    heatmap_matrix: torch.Tensor,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    xticks: dict = {},
    yticks: dict = {},
    title: Optional[str] = None,
    save_at: Optional[str] = None,
    center: Optional[float] = None,
    cbar_kws: dict = {},
    grid: dict = {"b": False},
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    annot: Optional[np.array] = None,
    mask: Optional[np.array] = None,
    fmt: Optional[str] = "",
    figsize: Optional[tuple] = None,
    square: bool = False,
):
    """
    Plots a heatmap for a heatmap matrix.

    Parameters
    ----------
    heatmap_matrix : torch.Tensor
        A matrix where each pixel represents the heat.
    xlabel : Optional[str]
        Set the label for the x-axis.
    ylabel : Optional[str]
        Set the label for the y-axis.
    xticks : dict
        Set the current tick locations and labels of the x-axis.
    yticks : dict
        Set the current tick locations and labels of the y-axis.
    title : Optional[str]
        Set a title for the axes.
    save_at : Optional[str]
        Save the current figure.
    center : Optional[float]
        The value at which to center the colormap when plotting divergant data.
        Using this parameter will change the default cmap if none is specified.
    cbar_kws : dict
        Keyword arguments for matplotlib.figure.Figure.colorbar().
    grid : dict
        Configure the grid lines.
    vmin : Optional[float]
        Value to anchor the colormap, otherwise it is inferred from the data and other keyword arguments.
    vmax : Optional[float]
        Value to anchor the colormap, otherwise it is inferred from the data and other keyword arguments.
    annot : Optional[np.array]
        If True, write the data value in each cell.
        If an array-like with the same shape as data, then use this to annotate the heatmap instead of the data.
        Note that DataFrames will match on position, not index.
    mask : Optional[np.array]
        If passed, data will not be shown in cells where mask is True.
        Cells with missing values are automatically masked.
    fmt : Optional[str]
        String formatting code to use when adding annotations.
    figsize : Optional[tuple]
        width, height in inches.
        If not provided, defaults to rcParams["figure.figsize"] = [6.4, 4.8].
    square : bool
        If True, set the Axes aspect to “equal” so each cell will be square-shaped.

    Returns
    -------
    None
    """

    plt.figure(figsize=figsize)
    fig = sns.heatmap(
        heatmap_matrix,
        center=center,
        cmap="coolwarm",
        robust=True,
        cbar_kws=cbar_kws,
        vmin=vmin,
        vmax=vmax,
        annot=annot,
        mask=mask,
        fmt=fmt,
        square=square,
    )
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.xticks(**xticks)
    plt.yticks(**yticks),
    plt.title(title)
    plt.grid(**grid)
    plt.show()
    if save_at:
        fig.figure.savefig(save_at + ".png")
