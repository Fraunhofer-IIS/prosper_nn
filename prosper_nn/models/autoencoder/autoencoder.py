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
from typing import Type


class Autoencoder(nn.Module):
    """
    An autoencoder that encodes an input into a smaller representation.
    Additionally the model trains a decoder to map the encoded representation back
    to the original dimension.
    """
    def __init__(
        self,
        n_inputs: int,
        n_hidden_neurons: int,
        activation: Type[torch.autograd.Function] = torch.tanh,
    ) -> None:
        """
        Parameters
        ----------
        n_inputs : int
            The number of features in one input sample.
        n_hidden_neurons : int
            The dimension the input should be encoded to.
        activation : Type[torch.autograd.Function]
            The activation function that is applied after the linear encoding.
        Returns
        -------
        None
        """
        super(Autoencoder, self).__init__()
        self.n_inputs = n_inputs
        self.n_hidden_neurons = n_hidden_neurons
        self.activation = activation

        self.encode = nn.Linear(n_inputs, n_hidden_neurons)
        self.decode = nn.Linear(n_hidden_neurons, n_inputs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            The input sample or a batch of input samples.

        Returns
        -------
        torch.Tensor
        """
        x = self.encode(x)
        compressed_state = self.activation(x)
        output = self.decode(compressed_state)
        return output
