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

import torch.nn as nn
import torch
import numbers
from typing import Type


class DeepFeedForward(nn.Module):
    """
    Create a Deep Feed Forward Neural Network
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        deepness: int,
        activation: Type[torch.autograd.Function] = None,
        dropout_rate: float = 0.0,
    ) -> None:
        """
        Parameters
        ----------
        input_dim : int
            The input dimension of the model.
        hidden_dim : int
            The size of the hidden layer. All hidden layers have the same dimension.
        output_dim : int
            The output dimension of the returned Deep-Feed-Forward Neural Network.
            It is equal for all levels of deepness.
        deepness : int
            The number of hidden layers in the Neural Network.
            It corresponds with the  amount of Feed Forward Neural Network paths to an output layer.
            A deepness equals one leads to a normal Feed-Forward Neural Network.
        activation : nn.functional, optional
            The activation function that is applied on the output of the hidden layers.
            The same function is used on all hidden layers.
            No function is applied if no function is given.
        dropout_rate : float, optional
            The rate to which input nodes of the model are set so zero during training of the model.
            The value of the dropout_rate should be in the range of [0, 1).
            If the value is zero or no value is given, no dropout is applied.

        Returns
        -------
        None
        """

        super(DeepFeedForward, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.deepness = deepness
        self.activation = activation
        self.dropout_rate = dropout_rate

        self._check_variables()

        # Define Layers
        for level in range(self.deepness):
            setattr(
                self,
                "hidden_layer%d" % level,
                nn.Linear(self.input_dim, self.hidden_dim),
            )

            if level < self.deepness:
                setattr(
                    self,
                    "hidden_layer_connector%d" % level,
                    nn.Linear(self.hidden_dim, self.hidden_dim),
                )

            setattr(
                self,
                "output_layer%d" % level,
                nn.Linear(self.hidden_dim, self.output_dim),
            )

    def forward(self, x: torch.Tensor) -> torch.tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            The input tensor given to the Deep-Feed-Forward Neural Network.
            It should have shape=(batchsize, input_dim).

        Returns
        -------
        torch.Tensor
            A output of a level in the model is stored in the tensor with the index that is equal to the level.
            Therefore the first dimension corresponds to the level of deepness in the model.
            The complete output shape is: shape=(deepness, batchsize, output_dim).
        """

        # Dropout on Input
        if self.dropout_rate > 0:
            x = nn.Dropout(self.dropout_rate)(x)
        # All Connections to Hidden Layers
        self.output_hidden_layer = self.deepness * [None]
        for level in range(self.deepness):
            self.horizontal_path = getattr(self, "hidden_layer%d" % level)(x)
            if level > 0:
                self.vertical_path = getattr(
                    self, "hidden_layer_connector%d" % (level - 1)
                )(self.output_hidden_layer[level - 1])
                self.output_hidden_layer[level] = (
                    self.horizontal_path + self.vertical_path
                )
            else:
                self.output_hidden_layer[level] = self.horizontal_path
            if self.activation is not None:
                self.output_hidden_layer[level] = self.activation(
                    self.output_hidden_layer[level]
                )

        # All Connections to Output Layers
        self.output_output_layer = torch.empty(self.deepness, x.size(0), 1)

        for level in range(self.deepness):
            self.horizontal_path = getattr(self, "output_layer%d" % level)(
                self.output_hidden_layer[level]
            )
            if level > 0:
                self.vertical_path = (self.output_output_layer[level - 1]).detach()
                self.output_output_layer[level] = (
                    self.horizontal_path + self.vertical_path
                )
            else:
                self.output_output_layer[level] = self.horizontal_path

        return self.output_output_layer

    def _check_variables(self) -> None:
        """
        Checks if self.input_dim, self.hidden_dim, self.output_dim,
        self.deepness, self.dropout_rate have valid inputs.
        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        if not isinstance(self.input_dim, int) or not 0 < self.input_dim:
            raise ValueError(
                "input_dim should be an integer in range [1, inf) "
                "representing the input dimension"
            )
        if not isinstance(self.hidden_dim, int) or not 0 < self.hidden_dim:
            raise ValueError(
                "hidden_dim should be an integer in range [1, inf) "
                "representing the dimension of the hidden layer"
            )
        if not isinstance(self.output_dim, int) or not 0 < self.output_dim:
            raise ValueError(
                "output_dim should be an integer in range [1, inf) "
                "representing the output dimension"
            )
        if not isinstance(self.deepness, int) or not 0 < self.deepness:
            raise ValueError(
                "deepness should be an integer in range [1, inf) "
                "representing the deepness of the neural network"
            )
        if (
            not isinstance(self.dropout_rate, numbers.Number)
            or not 0 <= self.dropout_rate <= 1
            or isinstance(self.dropout_rate, bool)
        ):
            raise ValueError(
                "dropout_rate should be a number in range [0, 1] "
                "representing the probability of an element being "
                "zeroed"
            )
