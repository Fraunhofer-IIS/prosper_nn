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


class ECNN_LSTM_Cell(nn.Module):

    """
    LSTM cell of a Error-Correction Neural Network (ECNN).
    Compared to the normal ECNN cell, $s_t$ is calculated differently:

    .. math::
        s_t = (1- LSTMregulator)s_{t-1} + LSTMregulator(tanh(As_{t-1} + Bu_t))

    The LSTM regulator is a diagonal matrix with diagonal entries between 0 and 1. At the start of learning, they are
    all set to one, which means, that the architecture defaults to the regular ECNN architecture. The other extreme
    would be for the LSTM regulator to only have entries which are 0. Then $s_t=s_{t-1}$ and there is total
    long-term memory.
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
        activation : nn.functional, optional
            The activation function that is applied on the output of the hidden layers.
            The same function is used on all hidden layers.
        n_features_Y: int
            The number of outputs, i.e. the number of elements of Y at each time
            step. The default is 1.


        Returns
        -------
        None

        """
        super(ECNN_LSTM_Cell, self).__init__()

        self.A = nn.Linear(n_state_neurons, n_state_neurons, bias=False)
        self.B = nn.Linear(n_features_U, n_state_neurons, bias=False)
        self.C = nn.Linear(n_state_neurons, n_features_Y, bias=False)
        self.D = nn.Linear(n_features_Y, n_state_neurons, bias=False)
        self.LSTM_regulator = nn.Linear(1, n_state_neurons, bias=False)
        nn.init.ones_(self.LSTM_regulator.weight)

        self.act = activation

        self.n_features_Y = n_features_Y

    def forward(
        self, state: torch.Tensor, U: torch.Tensor = None, Y: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the ECNN cell.

        Parameters
        ----------
        U : torch.Tensor
            The input for the ECNN at time t.
            U should have shape=(batchsize, n_features_U).
        Y : torch.Tensor
            The output of the ECNN at time t-1.
            Y should have shape=(batchsize, 1).
            The Y of the last time step is used to calculate
            the error for the error-correction.
        state : torch.Tensor
            The hidden state of the ECNN at time t-1.
            state should have shape=(batchsize, n_state_neurons).

        Returns
        -------
        tuple
            Contains output, which is the error at time t-1 and has the same
            dimensions as Y, and state, which is the hidden state at time t.
        """

        expectation = self.C(state)

        self.LSTM_regulator_matrix = torch.diag_embed(
            torch.reshape(self.LSTM_regulator.weight, (-1,))
        )
        if U is not None:
            if Y is not None:
                output = expectation - Y
                state = (
                    state
                    + (self.act(self.A(state) + self.B(U) + self.D(output)) - state)
                    @ self.LSTM_regulator_matrix
                )
            else:
                output = expectation
                state = (
                    state
                    + (self.act(self.A(state) + self.B(U)) - state)
                    @ self.LSTM_regulator_matrix
                )
        else:
            if Y is not None:
                output = expectation - Y
                state = (
                    state
                    + (self.act(self.A(state) + self.D(output)) - state)
                    @ self.LSTM_regulator_matrix
                )
            else:
                output = expectation
                state = (
                    state
                    + (self.act(self.A(state)) - state) @ self.LSTM_regulator_matrix
                )

        return (output, state)
