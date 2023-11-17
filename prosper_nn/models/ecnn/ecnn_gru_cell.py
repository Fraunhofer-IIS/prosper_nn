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


class ECNN_GRU_3_variant(nn.Module):

    """
    ECNN_GRU_3_variant cell of a Error-Correction Neural Network (ECNN).
    Compared to the normal ECNN cell, $s_t$ is calculated differently:

    .. math::
        s_t = (1 - \sigma(update\_vector)) \\circ s_{t-1} + \\sigma(update\_vector) \\circ (tanh(As_{t-1} + Bu_t + D(\\hat{y}_{t-1} - y_{t-1})))

    The implementation is similar to version 3 of the GRU variants in the following paper.
    One difference is that $$r_t$$ is fixed to a vector with ones in our implementation.

    R. Dey and F. M. Salem, "Gate-variants of Gated Recurrent Unit (GRU) neural networks,"
    2017 IEEE 60th International Midwest Symposium on Circuits and Systems (MWSCAS),
    Boston, MA, USA, 2017, pp. 1597-1600, doi: 10.1109/MWSCAS.2017.8053243  [Titel anhand dieser DOI in Citavi-Projekt übernehmen]

    If the update vector has large values, the sigmoid function converges toward 1 and
    the architecture defaults to the regular ECNN archictecture.
    On the other hand, if the update vector contains large negative values, then
    $s_t=s_{t-1}$ and there is total memory.

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
        super(ECNN_GRU_3_variant, self).__init__()

        self.A = nn.Linear(n_state_neurons, n_state_neurons, bias=False)
        self.B = nn.Linear(n_features_U, n_state_neurons, bias=False)
        self.C = nn.Linear(n_state_neurons, n_features_Y, bias=False)
        self.D = nn.Linear(n_features_Y, n_state_neurons, bias=False)

        self.update_vector = nn.Parameter(
            torch.zeros(n_state_neurons), requires_grad=True
        )

        self.ones = nn.Parameter(
            torch.ones_like(self.update_vector), requires_grad=False
        )

        self.act = activation

        self.n_features_Y = n_features_Y

    def forward(
        self, state: torch.Tensor, U: torch.Tensor = None, Y: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the ECNN GRU 3 variant.

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

        if Y is not None:
            output = expectation - Y
            error_correction = self.D(output)
        else:
            output = expectation
            error_correction = 0

        if U is not None:
            candidate_activation = self.act(
                self.A(state) + self.B(U) + error_correction
            )
        else:
            candidate_activation = self.act(self.A(state) + error_correction)

        state = (self.ones - torch.sigmoid(self.update_vector)) * state + torch.sigmoid(
            self.update_vector
        ) * candidate_activation

        return output, state
