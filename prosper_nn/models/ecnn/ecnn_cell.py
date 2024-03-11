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
from .gru_cell_variant import GRU_3_variant
from typing import Tuple, Union, Optional


class ECNNCell(nn.Module):
    """
    Cell of a Error-Correction Neural Network (ECNN).
    It models one time step of an ECNN:

    .. math::
        s_t = tanh(As_{t-1} + Bu_t + D(\hat{y}_{t-1} - y_{t-1}))
        \hat{y}_t = Cs_t

    """

    def __init__(
        self,
        n_features_U: int,
        n_state_neurons: int,
        n_features_Y: int = 1,
        recurrent_cell_type: str = "elman",
        kwargs_recurrent_cell: dict = {},
    ):
        """
        Parameters
        ----------
        n_features_U: int
            The number of inputs, i.e. the number of elements of U at each time
            step.
        n_state_neurons: int
            The number of neurons of the hidden layer, i.e. the hidden state state
            at each time step.
        n_features_Y: int
            The number of outputs, i.e. the number of elements of Y at each time
            step. The default is 1.
        recurrent_cell_type: str
            Select the cell for the state transition. The cells elman, lstm, gru
            (all from pytorch) and gru_3_variant (from prosper_nn) are supported.
        kwargs_recurrent_cell: dict
            Parameters for the recurrent cell. Activation function can be set here.

        Returns
        -------
        None

        """
        super(ECNNCell, self).__init__()

        self.C = nn.Linear(n_state_neurons, n_features_Y, bias=False)

        if recurrent_cell_type == "elman":
            self.recurrent_cell = nn.RNNCell
        elif recurrent_cell_type == "lstm":
            self.recurrent_cell = nn.LSTMCell
        elif recurrent_cell_type == "gru":
            self.recurrent_cell = nn.GRUCell
        elif recurrent_cell_type == "gru_3_variant":
            self.recurrent_cell = GRU_3_variant
        else:
            raise ValueError(
                f"recurrent_cell_type: {recurrent_cell_type} is not known."
                "Choose from elman, lstm, gru or gru_3_variant."
            )
        self.recurrent_cell = self.recurrent_cell(
            input_size=n_features_U + n_features_Y,
            hidden_size=n_state_neurons,
            **kwargs_recurrent_cell,
        )

        self.n_features_Y = n_features_Y
        self.n_features_U = n_features_U

    def forward(
        self,
        state: Union[torch.Tensor, Tuple[torch.Tensor]],
        U: Optional[torch.Tensor] = None,
        Y: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculates one time step with the inputs and returns the prediction and the state
        of the next time step.

        Parameters
        ----------
        state : torch.Tensor
            The hidden state of the ECNN at time t-1.
            state should have shape=(batchsize, n_state_neurons).
        U : torch.Tensor
            The input for the ECNN at time t (if known).
            U should have shape=(batchsize, n_features_U).
        Y : torch.Tensor
            The output of the ECNN at time t-1.
            Y should have shape=(batchsize, n_features_Y).
            The Y of the last time step (if known) is used to calculate
            the error for the error-correction.

        Returns
        -------
        tuple
            Contains output, which is the error or the forecast at time t-1
            and has the same dimensions as Y,
            and state, which is the hidden state at time t.
        """
        if isinstance(state, Tuple):
            expectation = self.C(state[0])
        else:
            expectation = self.C(state)

        device = expectation.device
        batchsize = expectation.shape[0]

        if Y is not None:
            output = expectation - Y
            error_correction = output
        else:
            output = expectation
            error_correction = torch.zeros(
                (batchsize, self.n_features_Y), device=device
            )

        if U is None:
            U = torch.zeros((batchsize, self.n_features_U), device=device)

        input = torch.cat((U, error_correction), dim=-1)
        state = self.recurrent_cell(input, state)
        return output, state

    def get_batchsize(self, state):
        if isinstance(state, Tuple):
            return state[0].shape[0]
        else:
            return state.shape[0]