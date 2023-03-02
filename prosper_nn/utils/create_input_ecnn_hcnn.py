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
from typing import Tuple, Optional
import warnings


def create_input(
    Y: torch.Tensor,
    past_horizon: int,
    batchsize: int,
    U: Optional[torch.Tensor] = None,
    future_U: Optional[bool] = False,
    forecast_horizon: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Takes time series of target variable Y and (optional) external variables U and
    reshapes them to be used for ecnn or hcnn or any model based on them.

    Parameters
    ----------
    Y: torch.Tensor
        The time series of the target variable. Should be of
        shape=(length of time series, n_features_Y).
    past_horizon: int
        The sequence length that is meant to be used for predictions.
    batchsize: int
    U: Optional[torch.Tensor]
        The time series of the inputs (external/explanatory variables). Should
        be of shape=(length of time series, n_features_U).
        If no U is given, the input for a hcnn model is created.
    future_U: boolean
        Is U also known in the future? Then a sequence of U should have length
        past_horizon+forecast_horizon, else past_horizon.
    forecast_horizon: int
        If future_U is true, this is used to create U_batches.

    Returns
    -------
    Y_batches: torch.Tensor
        Reshaped target variable that can be used for ECNNs or HCNNs.
        shape=(n_seqs, past_horizon, batchsize, n_features_Y).
    U_batches: torch.Tensor
        Only if U has been passed.
        Reshaped inputs that can be used for ECNNs.
        shape=(n_seqs, past_horizon, batchsize, n_features_U) or
        shape=(n_seqs, past_horizon+forecast_horizon, batchsize, n_features_U),
        depending on future_U.

    """

    # check requirements for U, future_U, forecast_horizon, Y
    if U is not None:
        if future_U and (forecast_horizon is None):
            raise ValueError(
                "If future U shall be used, the forecast horizon " "has to be given."
            )
        if future_U:
            if Y.shape[0] > (U.shape[0] - forecast_horizon):
                length_diff = Y.shape[0] - U.shape[0]
                Y = Y[0 : -(forecast_horizon - length_diff)]
                warnings.warn(
                    "For the last values of Y there are not enough "
                    "future Us, so they will be discarded."
                )

        elif Y.shape[0] != U.shape[0]:
            raise ValueError("Y and U have to be the same length.")

    # check if number of sequences is multiple of batchsize
    n_seqs = Y.shape[0] - past_horizon + 1
    if n_seqs % batchsize != 0:
        warnings.warn(
            "The number of sequences generated from the data are not a multiple of batchsize. "
            "The first %s "
            "sequences will be discarded." % (n_seqs % batchsize),
            stacklevel=2,
        )
        Y = Y[n_seqs % batchsize :]
        if U is not None:
            U = U[n_seqs % batchsize :]
        n_seqs = Y.shape[0] - past_horizon + 1

    Y_sequences = Y.unfold(0, past_horizon, 1).permute(2, 0, 1)
    Y_batches = torch.stack(torch.split(Y_sequences, batchsize, dim=1))

    if U is not None:
        if future_U:
            window_size = past_horizon + forecast_horizon
        else:
            window_size = past_horizon
        U_sequences = U.unfold(0, window_size, 1).permute(2, 0, 1)
        U_batches = torch.stack(torch.split(U_sequences, batchsize, dim=1))
        return Y_batches, U_batches

    return Y_batches
