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

import matplotlib.pyplot as plt
import torch
import numpy as np
from scipy.interpolate import interp1d
from typing import Optional, List

from . import visualization


def plot_time_series(
    expected_time_series: torch.tensor,
    target: Optional[torch.Tensor] = None,
    uncertainty: Optional[torch.Tensor] = None,
    save_at: Optional[str] = None,
    xlabel: str = "Time steps",
    ylabel: str = "Output",
    title: str = "Expected time series/Target comparison",
    labels: Optional[List[str]] = None,
) -> None:
    """
    Plot time series in line-style in a nice way.
    There are different possibilities what should be included in the plot.
    It is necessary to plot at least one time series (expected_time_series),
    but it is optional if a target time series and/or
    the uncertainty of the time series should be included in the plot.

    Parameters
    ----------
    expected_time_series : torch.tensor
        A one dimensional torch tensor which represents a time series.
        For example this can be the result of a forecast.
    target : torch.Tensor
        The variable is an one dimensional torch tensor that represents the target line of a forecast.
        The target is optional. If no target is given it is not included in the plot.
    uncertainty : torch.Tensor
        The uncertainty variable is a multidimensional torch tensor.
        Each column should represent one time series. If an ensemble model is calculated,
        the single forecasts from the models included in the ensemble can be seen as uncertainty of
        the mean output. Therefore, all the forecasts of the single models are plotted to the figure
        if a variable is given.
    save_at : str
        If defined the plot is saved as a png in the given file name.
    xlabel : str
        The label of the x axis in the plot.
    ylabel : str
        The label of the y axis in the plot.
    title : str
        The title that is shown in the plot.
    labels : List[str]
        If labels are given they are used for describing the plotted curves.
        Define one or two labels depending if target is defined or not.
        For uncertainty no label can be given.

    Returns
    -------
    None
    """
    fig, ax = plt.subplots()
    if target is not None:
        if len(target) > len(expected_time_series):
            diff = len(target) - len(expected_time_series)
            expected_time_series = expected_time_series.numpy()
            expected_time_series = np.append(expected_time_series, [np.nan] * diff)
        if len(target) < len(expected_time_series):
            diff = len(expected_time_series) - len(target)
            target = target.numpy()
            target = np.append(target, [np.nan] * diff)
    t = range(1, len(expected_time_series) + 1)
    if uncertainty is not None:
        if uncertainty.T.shape[1] > len(expected_time_series):
            raise ValueError("Uncertainty is longer than expected_time_series.")
        if uncertainty.T.shape[1] < len(expected_time_series):
            diff = len(expected_time_series) - uncertainty.shape[0]
            to_append = np.array([[np.nan] * uncertainty.shape[1]] * diff)
            uncertainty = uncertainty.numpy()
            uncertainty = np.append(to_append, uncertainty, axis=0)

        ax.plot(t, uncertainty, color="grey", alpha=0.25, label="_nolegend_")
    ax.plot(t, expected_time_series)

    if target is not None:
        ax.plot(t, target)
        if not labels:
            labels = ["Expected time series", "Target"]
        assert len(labels) == 2, "There are two labels needed"
    else:
        if not labels:
            labels = ["Expected time series"]
        assert len(labels) == 1, "There is one label needed"

    ax.set(
        xlabel=xlabel,
        ylabel=ylabel,
        title=title,
        xticks=range(1, len(expected_time_series) + 1, 2),
    )
    ax.legend(labels)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid()

    if save_at:
        fig.savefig(save_at + ".png")
    plt.show()


# Function Definitions for Heatmap of Forecasts


def _gaussian(x: torch.Tensor, mean: torch.Tensor, sigma: float) -> float:
    """
    Calculates the result of the gaussian function.

    Parameters
    ----------
    x : torch.Tensor
        Input on which the function should be applied. The tensor contains one float.
    mean : torch.Tensor
        Mean value of the gaussian function. The tensor contains one float.
    sigma : float
        Is a variable for the width of the gaussian bump.

    Returns
    -------
    float
        Output of the gaussian function for a given input.
    """
    return torch.exp(-torch.pow((x - mean) / sigma, 2.0))


def _interpolate_forecasts(forecasts: torch.Tensor, window_width: int) -> torch.Tensor:
    """
    Interpolates the forecasts in the way that the time series has window_width
    time steps after the interpolation.

    Parameters
    ----------
    forecasts : torch.Tensor
        A multidimensional torch tensor. Each row of the tensor contains a forecast.
    window_width : int
        The wished total amount of points in a forecast after interpolation.

    Returns
    -------
    torch.Tensor
        Linear interpolated forecasts.

    """

    forecast_horizon = forecasts.shape[1]

    x = torch.linspace(1, forecast_horizon, steps=forecast_horizon)
    interpolated_forecasts = torch.empty((0,))
    for forecast in forecasts:
        interpolated = torch.tensor(
            interp1d(x, forecast)(
                torch.linspace(1, forecast_horizon, steps=window_width)
            )
        )
        interpolated_forecasts = torch.cat(
            [interpolated_forecasts, interpolated.reshape(1, -1)], dim=0
        )
    return interpolated_forecasts


def _calculate_heatmap_matrix(
    interpolated_forecasts: torch.Tensor,
    sigma: float,
    window_width: int,
    window_height: int,
) -> torch.Tensor:
    """
    For each entry in a matrix, with shape = (window_height, window_width),
    a heat is calculated. The heat represents how many and how close the
    time series in interpolated_forecasts are to the entry.

    Parameters
    ----------
    interpolated_forecasts : torch.Tensor
        A multidimensional torch tensor. Each row of the tensor contains a forecast.
    sigma : float
        Sigma is a parameter in the gaussian function and influences the width of the gaussian bump.
        A higher value for sigma creates more blurred forecasts in the heatmap.
    window_width : int
        The width (amount of pixels) of the heatmap matrix
    window_height : int
        The height (amount of pixels) of the heatmap matrix

    Returns
    -------
    torch.tensor
        A matrix where each pixel represents the heat of a forecast value at a specific time.

    """
    heatmap = torch.empty((window_height, window_width))
    y_positions = torch.linspace(
        torch.min(interpolated_forecasts),
        torch.max(interpolated_forecasts),
        window_height,
    )
    for x_index in range(window_width):
        for y_index in range(window_height):
            forecasts_at_x_index = [
                forecast[x_index] for forecast in interpolated_forecasts
            ]
            heat = sum(
                [
                    _gaussian(y_forecast, y_positions[y_index], sigma)
                    for y_forecast in forecasts_at_x_index
                ]
            )
            heatmap[window_height - (y_index + 1)][x_index] = heat
    return heatmap


def heatmap_forecasts(
    forecasts: torch.Tensor,
    sigma: float = 0.05,
    num_interp: int = 15,
    window_height: int = 100,
    batch_first: bool = False,
    save_at: Optional[str] = None,
    xlabel: str = "Forecast Horizon",
    ylabel: str = "Values",
    title: str = "Heatmap Ensemble Forecast",
) -> None:
    """
    The function first interpolates values between the timesteps of the forecasts.
    Then for each pixel a heat is calculated with a gauss function and the
    result is plotted as a heatmap figure.
    The heat represents how many and how close the time series in forecasts are to the entry.
    At the end the heatmap is plotted.

    Parameters
    ----------
    forecasts : torch.tensor
        A multidimensional torch tensor. Each row of the tensor contains a forecast.
    sigma : float
        When the heat of a pixel is computed, a gaussian function is applied.
        The variable sigma is a value for the width of the gauss curve.
    num_interp : int
        The amount of points, which get interpolated between two timesteps in a forecast.
    window_height : int
        The amount of pixels in the height of the heatmap.
    batch_first:
        This is only necessary for recurrent neural networks when in the
        networks batch_first=True.
    save_at : str
        If defined the plot is saved as a png in the given file name.
    xlabel : str
        The label of the x axis in the plot.
    ylabel : str
        The label of the y axis in the plot.
    title : str
        The title that is shown in the plot.

    Returns
    -------
    None
    """

    if not batch_first:
        forecasts = forecasts.permute(1, 0)
    window_width = num_interp * (forecasts.shape[1] - 1)
    interpolated_forecasts = _interpolate_forecasts(forecasts, window_width)
    heatmap_matrix = _calculate_heatmap_matrix(
        interpolated_forecasts, sigma, window_width, window_height
    )
    heatmap_matrix = torch.div(heatmap_matrix, heatmap_matrix.max(axis=0)[0])
    yticks = {
        "ticks": range(0, window_height + 1, 10),
        "labels": [
            x.round(2)
            for x in np.linspace(torch.max(forecasts), torch.min(forecasts), num=11)
        ],
    }
    xticks = {
        "ticks": range(0, window_width + 1, num_interp * 2),
        "labels": range(1, forecasts.shape[1] + 1, 2),
        "rotation": 0,
    }

    visualization.plot_heatmap(
        heatmap_matrix=heatmap_matrix,
        save_at=save_at,
        xlabel=xlabel,
        ylabel=ylabel,
        title=title,
        yticks=yticks,
        xticks=xticks,
        grid={"linestyle": "--", "alpha": 0.5},
        figsize=(10, 5),
    )
