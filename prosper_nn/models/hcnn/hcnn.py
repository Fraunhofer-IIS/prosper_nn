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
from typing import Optional, Type
from . import hcnn_cell, hcnn_lstm_cell


class HCNN(nn.Module):
    """
    The HCNN class creates a Historical Consistent Neural Network.

    A Historical Consistent Neural Network belongs to the class of Recurrent Neural Networks.
    A special feature of the architecture is that it has no input in the common sense.
    Instead, all the inputs are also interpreted as targets that are equally important.
    So the architecture does not distinguish between input and output features.
    Furthermore, it uses teacher forcing to correct the hidden state.
    """

    def __init__(
        self,
        n_state_neurons: int,
        n_features_Y: int,
        past_horizon: int,
        forecast_horizon: int,
        sparsity: float = 0.0,
        activation: Type[torch.autograd.Function] = torch.tanh,
        lstm: bool = False,
        init_state: Optional[torch.Tensor] = None,
        learn_init_state: bool = True,
        teacher_forcing: float = 1,
        decrease_teacher_forcing: float = 0,
        backward_full_Y: bool = True,
        ptf_in_backward: bool = True,
    ):
        """
        Parameters
        ----------
        n_state_neurons : int
            The dimension of the state in the HCNN Cell. It must be a positive integer
            with n_state_neurons >= n_features_Y.
        n_features_Y : int
            The size of the data in each timestamp. It must be a positive integer.
        past_horizon : int
            The past horizon gives the amount of time steps into the past,
            where an observation is available.
            It represents the number of comparisons between expectation and observation and
            therefore the amount of teacher forcing.
        forecast_horizon : int
            The forecast horizon gives the amount of time steps into the future,
            where no observation is available.
            It represents the amount of forecast steps the model returns.
        sparsity : float
            The share of weights that are set to zero in the matrix A.
            These weights are not trainable and therefore always zero.
            For big matrices (dimension > 50) this can be necessary to guarantee
            numerical stability and it increases the long-term memory of the model.
        activation : Type[torch.autograd.Function]
            The activation function that is applied on the output of the hidden layers.
            The same function is used on all hidden layers.
            No function is applied if no function is given.
        lstm: boolean
           Include long short-term memory or not.
        init_state : torch.Tensor
            The initial state of the HCNN model.
            Can be given optionally and is chosen randomly if not specified.
        learn_init_state: boolean
            Learn the initial hidden state or not.
        teacher_forcing: float
            The probability that teacher forcing is applied for a single state neuron.
            In each time step this is repeated and therefore enforces stochastic learning
            if the value is smaller than 1.
        decrease_teacher_forcing: float
            The amount by which teacher_forcing is decreased each epoch.

        Returns
        -------
        None
        """
        super(HCNN, self).__init__()
        self.n_state_neurons = n_state_neurons
        self.n_features_Y = n_features_Y
        self.past_horizon = past_horizon
        self.forecast_horizon = forecast_horizon
        self.sparsity = sparsity
        self.activation = activation
        self.lstm = lstm
        self.teacher_forcing = teacher_forcing
        self.decrease_teacher_forcing = decrease_teacher_forcing
        self.backward_full_Y = backward_full_Y
        self.ptf_in_backward = ptf_in_backward
        self.state = [torch.tensor for _ in range(past_horizon + forecast_horizon + 1)]

        self._check_variables()

        self.init_state = nn.Parameter(
            torch.randn(1, n_state_neurons), requires_grad=learn_init_state
        )
        if init_state is not None:
            self.init_state.data = init_state

        # choose HCNNCell
        if lstm:
            self.HCNNCell = hcnn_lstm_cell.HCNN_LSTM_Cell
        else:
            self.HCNNCell = hcnn_cell.HCNNCell
        # init HCNNCELL
        self.HCNNCell = self.HCNNCell(
            self.n_state_neurons,
            self.n_features_Y,
            self.sparsity,
            self.activation,
            self.teacher_forcing,
            self.backward_full_Y,
            self.ptf_in_backward,
        )

    def forward(self, Y: torch.Tensor):
        """
        Parameters
        ----------
        Y : torch.Tensor
            Y should be 3-dimensional with the shape = (past_horizon, batchsize, n_features_Y).
            This timeseries of observations is used for training the model in
            order to predict future observations.

        Returns
        -------
        torch.Tensor
            Contains past_error, the forecasting errors along the past_horizon
            where Y is known, and forecast, the forecast along the
            forecast_horizon. Both can be used for backpropagation.
            shape=(past_horizon+forecast_horizon, batchsize, n_features_Y)
        """

        self.state[0] = self.init_state

        # LSTM: Keep entries of diagonal matrix between 0 and 1.
        if self.lstm:
            diag_entries = torch.clamp(self.HCNNCell.LSTM_regulator.weight.data, 0, 1)
            self.HCNNCell.LSTM_regulator.weight.data = torch.nn.Parameter(diag_entries)

        self._check_sizes(Y)
        batchsize = Y.shape[1]

        # reset saved cell outputs
        past_error = torch.zeros(self.past_horizon, batchsize, self.n_features_Y)
        forecast = torch.zeros(self.forecast_horizon, batchsize, self.n_features_Y)

        # past
        for t in range(self.past_horizon):
            if t == 0:
                self.state[t + 1], past_error[t] = self.HCNNCell(
                    self.state[t].repeat(batchsize, 1), Y[t]
                )
            else:
                self.state[t + 1], past_error[t] = self.HCNNCell(self.state[t], Y[t])
        # future
        for t in range(self.past_horizon, self.past_horizon + self.forecast_horizon):
            self.state[t + 1], forecast[t - self.past_horizon] = self.HCNNCell(
                self.state[t]
            )

        return torch.cat([past_error, forecast], dim=0)

    def adjust_teacher_forcing(self):
        """
        Decrease teacher_forcing each epoch by decrease_teacher_forcing until it reaches zero.
        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        if self.training:
            new_teacher_forcing = max(
                0, self.teacher_forcing - self.decrease_teacher_forcing
            )
            self.teacher_forcing = new_teacher_forcing
            self.HCNNCell.set_teacher_forcing(new_teacher_forcing)

    def _check_sizes(self, Y: torch.Tensor) -> None:
        """
        Checks if Y has right dimensions.
        Parameters
        ----------
        Y : torch.Tensor
            Y should be 3-dimensional with the shape = (past_horizon, batchsize, n_features_Y).
            This timeseries of observations is used for training the model in
            order to predict future observations.

        Returns
        -------
        None
        """
        if len(Y.shape) != 3:
            raise ValueError(
                "The shape for a batch of observations should be "
                "shape = (past_horizon, batchsize, n_features_Y)"
            )

        if (Y.shape[0] != self.past_horizon) or (Y.shape[2] != self.n_features_Y):
            raise ValueError(
                "Y must be of the dimensions"
                " shape = (past_horizon, batchsize, n_features_Y)."
                " Have you initialized HCNN with the"
                " right parameters?"
            )

    def _check_variables(self) -> None:
        """
        Checks if self.n_state_neurons, self.n_features_Y, self.past_horizon,
        self.forecast_horizon, self.sparsity have valid inputs.
        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        if (self.n_state_neurons < 1) or (type(self.n_state_neurons) != int):
            raise ValueError(
                "{} is not a valid number for n_state_neurons. "
                "It must be an integer greater than 0.".format(self.n_state_neurons)
            )
        if (self.n_features_Y < 1) or (type(self.n_features_Y) != int):
            raise ValueError(
                "{} is not a valid number for n_features_Y. "
                "It must be an integer greater than 0.".format(self.n_features_Y)
            )
        if self.n_features_Y > self.n_state_neurons:
            raise ValueError(
                "{} is not a valid number for n_state_neurons. "
                "It must be greater than or equal to n_features_Y ({})."
                "".format(self.n_state_neurons, self.n_features_Y)
            )
        if (self.past_horizon < 1) or (type(self.past_horizon) != int):
            raise ValueError(
                "{} is not a valid number for past_horizon. "
                "It must be an integer greater than 0.".format(self.past_horizon)
            )
        if (self.forecast_horizon < 0) or (type(self.forecast_horizon) != int):
            raise ValueError(
                "{} is not a valid number for forecast_horizon. "
                "It must be an integer equal or greater than 0.".format(
                    self.forecast_horizon
                )
            )
        if (self.sparsity < 0) or (self.sparsity > 1):
            raise ValueError(
                "{} is not a valid number for sparsity. "
                "It must be a value in the interval [0, 1].".format(self.sparsity)
            )
        if (self.teacher_forcing < 0) or (self.teacher_forcing > 1):
            raise ValueError(
                "{} is not a valid number for teacher_forcing. "
                "It must be a value in the interval [0, 1].".format(
                    self.teacher_forcing
                )
            )
        if (self.decrease_teacher_forcing < 0) or (self.decrease_teacher_forcing > 1):
            raise ValueError(
                "{} is not a valid number for decrease_teacher_forcing. "
                "It must be a value in the interval [0, 1].".format(
                    self.decrease_teacher_forcing
                )
            )
