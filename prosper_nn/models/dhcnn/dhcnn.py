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
from typing import Optional
from ..hcnn import hcnn_cell, hcnn_lstm_cell


class DHCNN(nn.Module):
    """
    The DHCNN class creates a Deep Historical Consistent Neural Network.
    The model uses multiple HCNNs in different levels. The state from the lower
    level is passed to one upper level. The first level is a HCNN with LSTM implementation.
    """

    def __init__(
        self,
        n_state_neurons: int,
        n_features_Y: int,
        past_horizon: int,
        forecast_horizon: int,
        deepness: int,
        sparsity: float = 0.0,
        activation: torch = torch.tanh,
        init_state: Optional[torch.Tensor] = None,
        learn_init_state: bool = True,
        teacher_forcing: float = 1,
        decrease_teacher_forcing: float = 0,
    ):
        """
        Parameters
        ----------
        n_state_neurons : int
            The dimension of the state in the HCNN Cell. It must be a positive integer with
            n_state_neurons >= n_features_Y.
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
        deepness : int
            The number of stacked HCNNs in the Neural Network.
            A deepness equal to 1 leads to a normal Historical Consistent Neural Network with LSTM.
        sparsity : float
            The share of weights that are set to zero in the matrix A.
            These weights are not trainable and therefore always zero.
            For big matrices (dimension > 50) this can be necessary to guarantee
            numerical stability and it increases the long-term memory of the model.
        activation : torch
            The activation function that is applied on the output of the hidden layers.
            The same function is used on all hidden layers.
            No function is applied if no function is given.
        init_state : torch.Tensor
            The initial states of the HCNN model.
            Can be given optionally and is chosen randomly if not specified.
            If given, it should have the shape = (deepness, 1, n_state_neurons)
        learn_init_state: boolean
            Learn the initial hidden state or not.
        teacher_forcing: float
            The probability that teacher forcing is  applied for a single state neuron.
            In each time step this is repeated and therefore enforces stochastic learning
            if the value is smaller than 1.
        decrease_teacher_forcing: float
            The amount by which teacher_forcing is decreased each epoch.

        Returns
        -------
        None
        """
        super(DHCNN, self).__init__()
        self.n_state_neurons = n_state_neurons
        self.n_features_Y = n_features_Y
        self.past_horizon = past_horizon
        self.forecast_horizon = forecast_horizon
        self.deepness = deepness
        self.sparsity = sparsity
        self.activation = activation
        self.init_state = init_state

        self.teacher_forcing = teacher_forcing
        self.decrease_teacher_forcing = decrease_teacher_forcing

        self.state = [
            [torch.tensor for _ in range(past_horizon + forecast_horizon + 1)]
            for _ in range(self.deepness)
        ]
        self._check_variables()

        for i in range(self.deepness):
            self.init_state = nn.Parameter(
                torch.randn(self.deepness, 1, n_state_neurons),
                requires_grad=learn_init_state,
            )
            if init_state is not None:
                self.init_state.data = init_state

        for i in range(deepness):
            if i == 0:
                HCNNCell = hcnn_lstm_cell.HCNN_LSTM_Cell
            else:
                HCNNCell = hcnn_cell.HCNNCell
            setattr(
                self,
                "hcnn_cell_level%d" % i,
                HCNNCell(
                    self.n_state_neurons,
                    self.n_features_Y,
                    self.sparsity,
                    self.activation,
                ),
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
            Contains for each HCNN level: The past_error, i.e. the forecasting errors along the past_horizon
            where Y is known, and forecast, i.e. the forecast along the
            forecast_horizon. Both can be used for backpropagation.
            shape=(deepness, past_horizon+forecast_horizon, batchsize, n_features_Y)
        """
        self._check_sizes(Y)
        self.batchsize = Y.shape[1]

        if self.decrease_teacher_forcing > 0:
            self._adjust_teacher_forcing()

        # reset saved cell outputs
        past_error = torch.zeros(
            self.deepness, self.past_horizon, self.batchsize, self.n_features_Y
        )
        forecast = torch.zeros(
            self.deepness, self.forecast_horizon, self.batchsize, self.n_features_Y
        )

        # LSTM: Keep entries of diagonal matrix between 0 and 1.
        diag_entries = torch.clamp(
            self.hcnn_cell_level0.LSTM_regulator.weight.data, 0, 1
        )
        self.hcnn_cell_level0.LSTM_regulator.weight.data = torch.nn.Parameter(
            diag_entries
        )

        # past
        for i in range(self.deepness):
            horizontal_stream = self.init_state[i].repeat(self.batchsize, 1)
            for t in range(self.past_horizon):
                # state is the sum of state from left side and upstream state for higher levels
                if i > 0:
                    self.state[i][t] = torch.add(
                        horizontal_stream, self.state[i - 1][t]
                    )
                else:
                    self.state[i][t] = horizontal_stream

                horizontal_stream, past_error[i][t] = getattr(
                    self, "hcnn_cell_level%d" % i
                )(self.state[i][t], Y[t])
            # future
            for t in range(
                self.past_horizon, self.past_horizon + self.forecast_horizon
            ):
                if i > 0:
                    self.state[i][t] = torch.add(
                        horizontal_stream, self.state[i - 1][t]
                    )
                else:
                    self.state[i][t] = horizontal_stream
                horizontal_stream, forecast[i][t - self.past_horizon] = getattr(
                    self, "hcnn_cell_level%d" % i
                )(self.state[i][t])
            self.state[i][self.past_horizon + self.forecast_horizon] = horizontal_stream

        return torch.cat([past_error, forecast], dim=1)

    def _adjust_teacher_forcing(self):
        """
        Decrease teacher_forcing each epoch by decrease_teacher_forcing until it reaches zero.
        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        new_teacher_forcing = max(
            0, self.teacher_forcing - self.decrease_teacher_forcing
        )
        self.teacher_forcing = new_teacher_forcing
        for i in range(self.deepness):
            getattr(self, "hcnn_cell_level%d" % i).set_teacher_forcing(
                new_teacher_forcing
            )

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
