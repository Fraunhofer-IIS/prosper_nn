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
from ..hcnn.hcnn_cell import HCNNCell


class CRCNN(nn.Module):
    """
    The CRCNN class creates a Causal Retro-Causal Neural Network.

    It consists of a number of branches, each one an HCNN model (using the HCNNCell), which are alternately causal (going forward in time) and retro-causal (going backward in time). The forecast between the last retro causal and the last causal branch is used for the actual forecast, the others are for training only. All causal branches use the same initial state (in the past) and the same state matrix A for making one step forward into the future. All the retro-causal branches use the same initial state (in the future) and the same state matrix A' for making one step backward into the past.

    When the errors are trained down to zero, the model converges to a CRCNN only containing one causal and one retro-causal branch. In contrast to the HCNN, the expectation at each time step is the sum of the outputs of one retro-causal and one causal branch. By this, hopefully both causal and retro-causal dynamics in the data are captured.
    """

    def __init__(
        self,
        n_state_neurons: int,
        n_features_Y: int,
        past_horizon: int,
        forecast_horizon: int,
        n_branches: int,
        batchsize: int,
        sparsity: float = 0.0,
        activation: Type[torch.autograd.Function] = torch.tanh,
        init_state_causal: Optional[torch.Tensor] = None,
        learn_init_state_causal: bool = True,
        init_state_retro_causal: Optional[torch.Tensor] = None,
        learn_init_state_retro_causal: bool = True,
        teacher_forcing: float = 1,
        decrease_teacher_forcing: float = 0,
        mirroring: bool = False,
        no_ptf_mirror: bool = True,
    ):
        """
        Parameters
        ----------
        n_state_neurons : int
            The dimension of the state in the CRCNN Cell. It must be a positive integer.
        n_features_Y : int
            The size of the data in each time step. It must be a positive integer
            and n_features_Y <= n_state_neurons.
        past_horizon : int
            The past horizon gives the amount of time steps into the past where an observation is available.
        forecast_horizon : int
            The forecast horizon gives the amount of time steps into the future, where no observation is available.
            It represents the amount of forecast steps the model returns.
        n_branches : int
            The total number of branches of the CRCNN.  n_branches must be minimum 3 for teacher forcing to work.
        batchsize : int
            The amount of samples in each batch.
        sparsity : float
            The share of weights that are set to zero in the matrices A in the causal and the retro-causal cell.
            These weights are not trainable and therefore always zero. For big matrices (dimension > 50) this can be necessary to guarantee numerical stability and it increases the long-term memory of the model.
        activation : nn.functional, optional
            The activation function that is applied on the output of the hidden layers. The same function is used on all hidden layers. No function is applied if no function is given.
        init_state_causal : torch.Tensor
            The initial states of (all) the causal branches of the CRCNN model. Can be given optionally and is chosen randomly if not specified. If given, it should have the shape = (1, n_state_neurons).
        learn_init_state_causal: boolean
            Learn the initial hidden state of the causal branches or not.
        init_state_retro_causal : torch.Tensor
            The initial states of (all) the retro causal branches of the CRCNN model. Can be given optionally and is chosen randomly if not specified. If given, it should have the shape = (1, n_state_neurons).
        learn_init_state_retro_causal: boolean
            Learn the initial hidden state of the retro causal branches or not.
        teacher_forcing: float
            The probability that teacher forcing is applied for a single state neuron. In each time step this is repeated and therefore enforces stochastic learning if the value is smaller than 1.
        decrease_teacher_forcing: float
            The amount by which teacher_forcing is decreased each epoch.
        mirroring : bool
            If set to True, the mirror trick is applied. This means that a future_bias is added that learns the forecast and is used as a fake future Y to do teacher forcing even in the future. Even if the mirror trick is used for training, mirroring should be set to False when forecasting in order to get the real forecast and not the fake forecasting error (s. Returns).
        no_ptf_mirror : bool
            If mirroring is True and teacher_forcing < 1, the user can choose whether random teacher forcing is applied on the mirroring nodes or not. Therefore, it concerns the partial teacher forcing during the future.

        Returns
        -------
        None
        """

        super(CRCNN, self).__init__()
        self.n_state_neurons = n_state_neurons
        self.n_features_Y = n_features_Y
        self.past_horizon = past_horizon
        self.forecast_horizon = forecast_horizon
        self.n_branches = n_branches
        self.n_causal_branches = int((n_branches + 1) / 2)
        self.batchsize = batchsize
        self.sparsity = sparsity
        self.activation = activation
        self.teacher_forcing = teacher_forcing
        self.decrease_teacher_forcing = decrease_teacher_forcing
        self.mirroring = mirroring
        self.no_ptf_mirror = no_ptf_mirror

        self.state_causal = [
            [torch.tensor] * (past_horizon + forecast_horizon + 1)
            for _ in range(self.n_causal_branches)
        ]
        self.state_retro_causal = [
            [torch.tensor] * (past_horizon + forecast_horizon)
            for _ in range(self.n_causal_branches - 1)
        ]
        self._check_variables()

        self.init_state_causal = nn.Parameter(
            torch.randn(1, n_state_neurons), requires_grad=learn_init_state_causal
        )
        if init_state_causal is not None:
            self.init_state_causal.data = init_state_causal

        self.init_state_retro_causal = nn.Parameter(
            torch.randn(1, n_state_neurons), requires_grad=learn_init_state_retro_causal
        )
        if init_state_retro_causal is not None:
            self.init_state_retro_causal.data = init_state_retro_causal

        self.CRCNNCell_causal = HCNNCell(
            self.n_state_neurons,
            self.n_features_Y,
            self.sparsity,
            self.activation,
            self.teacher_forcing,
        )
        self.CRCNNCell_retro_causal = HCNNCell(
            self.n_state_neurons,
            self.n_features_Y,
            self.sparsity,
            self.activation,
            self.teacher_forcing,
        )

        self.future_bias = nn.Parameter(
            torch.zeros((self.forecast_horizon, self.batchsize, self.n_features_Y)),
            requires_grad=True,
        )

    def forward(self, Y: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        Y : torch.Tensor
            Y should be 3-dimensional with the shape = (past_horizon, batchsize, n_features_Y). This time series of observations is used for training the model in order to predict future observations.

        Returns
        -------
        torch.Tensor
            Contains past_error, i.e. the forecasting errors along the past_horizon where Y is known, and forecast, i.e. the forecast along the forecast_horizon, for each pair of causal and retro causal branches.
            If mirroring = True, the forecast contains the fake forecasting errors produced by using the future_bias as a fake future Y. In this case, therefore, the forecast should be used for training and the target should be 0.
            shape=(n_branches-1, past_horizon + forecast_horizon, batchsize, n_features_Y)
        """

        self._check_sizes(Y)

        if self.mirroring:
            future_bias = self.future_bias
        else:
            future_bias = [None] * self.forecast_horizon

        # reset saved cell outputs
        past_error = torch.zeros(
            self.n_branches - 1, self.past_horizon, self.batchsize, self.n_features_Y
        )
        forecast = torch.zeros(
            self.n_branches - 1,
            self.forecast_horizon,
            self.batchsize,
            self.n_features_Y,
        )

        # initialize causal and retro-causal branches
        for i in range(self.n_causal_branches):
            self.state_causal[i][0], _ = self.CRCNNCell_causal(
                self.init_state_causal.repeat(self.batchsize, 1)
            )
        for i in range(self.n_causal_branches - 1):
            self.state_retro_causal[i][-1], _ = self.CRCNNCell_retro_causal(
                self.init_state_retro_causal.repeat(self.batchsize, 1)
            )

        # First (causal) branch (no teacher-forcing)
        for t in range(self.past_horizon + self.forecast_horizon):
            self.state_causal[0][t + 1], _ = self.CRCNNCell_causal(
                self.state_causal[0][t]
            )

        # Pairs of retro-causal and causal branches
        for i in range(1, self.n_causal_branches):
            # retro-causal
            # future
            for t in range(
                self.past_horizon + self.forecast_horizon - 1, self.past_horizon - 1, -1
            ):
                if self.no_ptf_mirror:
                    self.CRCNNCell_retro_causal.set_teacher_forcing(1)
                (
                    self.state_retro_causal[i - 1][t - 1],
                    forecast[2 * i - 2, t - self.past_horizon],
                ) = self.CRCNNCell_retro_causal(
                    self.state_retro_causal[i - 1][t] + self.state_causal[i - 1][t],
                    future_bias[t - self.past_horizon],
                )
                if self.no_ptf_mirror:
                    self.CRCNNCell_retro_causal.set_teacher_forcing(
                        self.teacher_forcing
                    )

            # past
            for t in range(self.past_horizon - 1, 0, -1):
                (
                    self.state_retro_causal[i - 1][t - 1],
                    past_error[2 * i - 2, t],
                ) = self.CRCNNCell_retro_causal(
                    self.state_retro_causal[i - 1][t] + self.state_causal[i - 1][t],
                    Y[t],
                )

            past_error[2 * i - 2, 0] = (
                self.state_retro_causal[i - 1][t][:, : self.n_features_Y]
                + self.state_causal[i - 1][t][:, : self.n_features_Y]
                - Y[0]
            )

            # causal
            # past
            for t in range(self.past_horizon):
                (
                    self.state_causal[i][t + 1],
                    past_error[2 * i - 1, t],
                ) = self.CRCNNCell_causal(
                    self.state_causal[i][t] + self.state_retro_causal[i - 1][t], Y[t]
                )
            # future
            if self.no_ptf_mirror:
                self.CRCNNCell_causal.set_teacher_forcing(1)
            for t in range(
                self.past_horizon, self.past_horizon + self.forecast_horizon
            ):
                (
                    self.state_causal[i][t + 1],
                    forecast[2 * i - 1, t - self.past_horizon],
                ) = self.CRCNNCell_causal(
                    self.state_causal[i][t] + self.state_retro_causal[i - 1][t],
                    future_bias[t - self.past_horizon],
                )
            if self.no_ptf_mirror:
                self.CRCNNCell_causal.set_teacher_forcing(self.teacher_forcing)
        return torch.cat((past_error, forecast), dim=1)

    def _check_sizes(self, Y: torch.Tensor) -> None:
        """
        Checks if Y has right dimensions.
        Parameters
        ----------
        Y : torch.Tensor
            Y should be 3-dimensional with the shape = (past_horizon, batchsize, n_features_Y). This timeseries of observations is used for training the model in order to predict future observations.

        Returns
        -------
        None
        """

        if len(Y.shape) != 3:
            raise ValueError(
                "The shape for a batch of observations should be "
                "shape = (past_horizon, batchsize, n_features_Y)"
            )

        if Y.size() != torch.Size(
            (self.past_horizon, self.batchsize, self.n_features_Y)
        ):
            raise ValueError(
                "Y must be of the dimensions"
                " shape = (past_horizon, batchsize, n_features_Y)."
                " Have you initialized the CRCNN with the"
                " right parameters?"
            )

    def _check_variables(self) -> None:
        """
        Checks if self.n_state_neurons, self.n_features_Y, self.past_horizon,
        self.forecast_horizon, self.batchsize, self.sparsity have valid inputs.
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
        if (self.n_branches < 3) or (type(self.n_branches) != int):
            raise ValueError(
                "{} is not a valid number for n_branches. "
                "It must be an integer equal or greater than 3.".format(self.n_branches)
            )
        if (self.batchsize < 1) or (type(self.batchsize) != int):
            raise ValueError(
                "{} is not a valid number for batchsize. "
                "It must be an integer greater than 0.".format(self.batchsize)
            )
        if (self.sparsity < 0) or (self.sparsity > 1):
            raise ValueError(
                "{} is not a valid number for sparsity. "
                "It must be a value in the interval [0, 1].".format(self.sparsity)
            )
        if (self.decrease_teacher_forcing < 0) or (self.decrease_teacher_forcing > 1):
            raise ValueError(
                "{} is not a valid number for decrease_teacher_forcing. "
                "It must be a value in the interval [0, 1].".format(
                    self.decrease_teacher_forcing
                )
            )

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
            self.CRCNNCell_causal.set_teacher_forcing(new_teacher_forcing)
            self.CRCNNCell_retro_causal.set_teacher_forcing(new_teacher_forcing)
