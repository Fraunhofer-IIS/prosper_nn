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
from typing import Tuple, Type


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
        activation: Type[torch.autograd.Function] = torch.tanh,
        n_features_Y: int = 1,
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
        activation: nn.functional
            The activation function that is applied on the output of the hidden layers.
            The same function is used on all hidden layers.
        n_features_Y: int
            The number of outputs, i.e. the number of elements of Y at each time
            step. The default is 1.

        Returns
        -------
        None

        """
        super(ECNNCell, self).__init__()

        self.A = nn.Linear(n_state_neurons, n_state_neurons, bias=False)
        self.B = nn.Linear(n_features_U, n_state_neurons, bias=False)
        self.C = nn.Linear(n_state_neurons, n_features_Y, bias=False)
        self.D = nn.Linear(n_features_Y, n_state_neurons, bias=False)

        self.act = activation

        self.n_features_Y = n_features_Y

    def forward(
        self, state: torch.Tensor, U: torch.Tensor = None, Y: torch.Tensor = None
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

        expectation = self.C(state)

        if U is not None:
            if Y is not None:
                output = expectation - Y
                state = self.act(self.A(state) + self.B(U) + self.D(output))
            else:
                output = expectation
                state = self.act(self.A(state) + self.B(U))
        else:
            if Y is not None:
                output = expectation - Y
                state = self.act(self.A(state) + self.D(output))
            else:
                output = expectation
                state = self.act(self.A(state))

        return (output, state)
