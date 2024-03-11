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


class GRU_3_variant(nn.Module):

    """
    GRU_3_variant cell .
    State $s_t$ is calculated as:

    .. math::
        s_t = (1 - \sigma(update\_vector)) \\circ s_{t-1} + \\sigma(update\_vector) \\circ (tanh(As_{t-1} + Bx_t))

    The implementation is similar to version 3 of the GRU variants in the following paper.
    One difference is that $$r_t$$ is fixed to a vector with ones in our implementation.

    R. Dey and F. M. Salem, "Gate-variants of Gated Recurrent Unit (GRU) neural networks,"
    2017 IEEE 60th International Midwest Symposium on Circuits and Systems (MWSCAS),
    Boston, MA, USA, 2017, pp. 1597-1600, doi: 10.1109/MWSCAS.2017.8053243

    If the update vector has large values, the sigmoid function converges toward 1 and
    the architecture defaults to the regular RNNCell.
    On the other hand, if the update vector contains large negative values, then
    $s_t=s_{t-1}$ and there is total memory.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        activation: Type[torch.autograd.Function] = torch.tanh,
    ):
        """

        Parameters
        ----------
        input_size: int
            The number of inputs, i.e. the number of elements of input at each time
            step.
        hidden_size: int
            The number of neurons of the hidden layer, i.e. the hidden state state
            at each time step.
        activation : nn.functional, optional
            The activation function that is applied on the output of the hidden layers.
            The same function is used on all hidden layers.

        Returns
        -------
        None

        """
        super(GRU_3_variant, self).__init__()

        self.A = nn.Linear(hidden_size, hidden_size, bias=False)
        self.B = nn.Linear(input_size, hidden_size, bias=False)
        self.update_vector = nn.Parameter(torch.zeros(hidden_size), requires_grad=True)

        self.ones = nn.Parameter(
            torch.ones_like(self.update_vector), requires_grad=False
        )
        self.act = activation

    def forward(
        self,
        input: torch.Tensor,
        state: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the GRU 3 variant.

        Parameters
        ----------
        input : torch.Tensor
            The input for the cell at time t.
            It should have shape=(batchsize, input_size).
        state : torch.Tensor
            The hidden state at time t-1.
            state should have shape=(batchsize, hidden_size).

        Returns
        -------
        torch.Tensor
            Contains state, which is the hidden state for the next time step.
        """
        candidate_activation = self.act(self.A(state) + self.B(input))

        state = (self.ones - torch.sigmoid(self.update_vector)) * state + torch.sigmoid(
            self.update_vector
        ) * candidate_activation

        return state
