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

import numpy as np
import torch
import random
from typing import Tuple


def sample_data(
    sample_size: int, n_features_Y: int, n_features_U: int, signal_freq: int = 10
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Creates dummy time series with targets and exogenous data.
    Adaption from https://github.com/uselessskills/hcnn

    Parameters
    ----------
    sample_size: int
        The total length of the created time series.
    n_features_Y: int
        The number of target variables that should be created.
    n_features_U: int
        The number of exogenous features.
    signal_freq: int
        A number to set the frequence of the timeseries. Higher number creates more complex time series.

    Returns
    -------
    Y: torch.Tensor
        Created targets for time series with shape=(sample_size, n_features_Y).
    U: torch.Tensor
        Created exogenous features with shape=(sample_size, n_features_U).
    """

    def sine(X, signal_freq=60.0):
        return (
            np.sin(2 * np.pi * (X) / signal_freq)
            + np.sin(4 * np.pi * (X) / signal_freq)
        ) / 2.0

    def noisy(Y, noise_range=(-0.05, 0.05)):
        noise = np.random.uniform(noise_range[0], noise_range[1], size=Y.shape)
        return Y + noise

    def add_external(Y, U, noise_range=(-0.05, 0.05)):
        noise = np.random.uniform(noise_range[0], noise_range[1], size=Y.shape)
        external = torch.sum(U, dim=1) + noise
        return torch.Tensor(Y) + external

    random_offset = random.randint(0, sample_size)
    U = torch.rand(sample_size, n_features_U, dtype=torch.float32, requires_grad=False)
    X = np.arange(sample_size)

    Y = noisy(sine(X + random_offset, signal_freq))
    Y = add_external(Y, U)

    ys = [Y]
    for y in range(1, n_features_Y):
        y = noisy(sine(X + random_offset, signal_freq))
        y = add_external(y, U)
        ys.append(y)
    Y = torch.stack(ys, 1)

    return Y.float(), U
