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


class FFNN(nn.Module):
    """
    The Feed Forward Neural Network is a three layer network with a non-linearity
    in the hidden layer.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        activation: Type[torch.autograd.Function] = torch.tanh,
    ) -> None:
        """
        Parameters
        ----------
        input_dim : int
            The dimension of the input layer. It must be a positive integer.
        hidden_dim : int
            The dimension of the hidden layer. It must be a positive integer.
        output_dim : int
            The dimension of the output layer. It must be a positive integer.
        activation : torch
            The activation function that is applied on the output of the hidden layer.

        Returns
        -------
        None
        """
        super(FFNN, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.activation = activation

        self.hidden = nn.Linear(input_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            The input of the model.

        Returns
        -------
        torch.Tensor
            The output of the FFNN.
        """
        x = self.hidden(x)
        x = self.activation(x)
        return self.output(x)
