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
from . import ecnn_cell, ecnn_lstm_cell
from typing import Tuple, Type


class ECNN(torch.nn.Module):

    """
    The ECNN class creates a Error Correction Neural Network.

    An ECNN is an extension of an RNN where the forecast error at each time
    step is interpreted as a reaction to external influences unknown to the
    model. In order to correct the forecasting, this error is used as an
    additional input for the next time step, ideally substituting the
    unknown external information.

    The general architecture is given by

    .. math::
        s_t = tanh(As_{t-1} + Bu_t + D(\hat{y}_{t-1} - y_{t-1}))
        \hat{y}_t = Cs_t

    where $y_t$ is the target variable, $u_t$ is the explanatory feature and $s_t$ is the hidden state.
    $A$, $B$, $C$ and $D$ are matrices.

    For forecasting where neither $u_{t+2}$ nor $y_{t+1}$ is known, it's

    .. math::
        s_{t+2} = tanh(As_{t+1})
        \hat{y}_{t+2} = Cs_{t+2}

    These formulae are implemented in the ECNNCell class, which is used to construct this ECNN model.

    This implementation is based on
    Zimmermann HG., Neuneier R., Grothmann R. (2002) Modeling Dynamical Systems by Error Correction Neural Networks.
    In: Soofi A.S., Cao L. (eds) Modelling and Forecasting Financial Data. Studies in Computational Finance,
    vol 2. Springer, Boston, MA. https://doi.org/10.1007/978-1-4615-0931-8_12

    """

    def __init__(
        self,
        n_features_U: int,
        n_state_neurons: int,
        past_horizon: int,
        forecast_horizon: int = 1,
        lstm: bool = False,
        approach: str = "backward",
        init_state: torch.Tensor = None,
        learn_init_state: bool = True,
        activation: Type[torch.autograd.Function] = torch.tanh,
        n_features_Y: int = 1,
        future_U: bool = False,
    ) -> None:
        """
        Parameters
        ----------
        n_features_U: int
            The number of inputs, i.e. the number of elements of U at each time
            step.
        n_state_neurons: int
            The number of neurons of the hidden layer, i.e. the hidden state
            at each time step.
        past_horizon: int
            The length of the sequence of inputs and outputs used for
            prediction.
        forecast_horizon: int
            The forecast horizon.
        lstm: boolean
           Include long short-term memory or not. Use either ecnn_cell or ecnn_lstm_cell.
        approach: string
            Either "backward" or "forward".
            A backward approach means that the external features at time t
            have a direct impact on the hidden state at time t.
            A forward approach means that the external features at time t
            only have a direct impact on the hidden state at time t+1.
        init_state: torch.Tensor
            The initial hidden state. If none is given, it is generated
            randomly.
        learn_init_state: boolean
            Learn the initial hidden state or not.
        activation : nn.functional
            The activation function that is applied on the output of the hidden layers.
            The same function is used on all hidden layers.
        n_features_Y: int
            The number of outputs, i.e. the number of elements of Y at each time
            step. The default is 1.
        future_U: boolean
            If false, U is assumed to be only known in the past and thus have
            the length past_horizon.
            If true, U is assumed to be also known in the future, e.g. weekdays,
            and thus have the length past_horizon+forecast_horizon.

        Returns
        -------
        None
        """

        super(ECNN, self).__init__()

        self.n_features_U = n_features_U
        self.n_features_Y = n_features_Y
        self.n_state_neurons = n_state_neurons
        self.past_horizon = past_horizon
        self.forecast_horizon = forecast_horizon
        self.lstm = lstm
        self.approach = approach
        self.future_U = future_U

        self._check_variables()

        self.state = [torch.tensor] * (past_horizon + forecast_horizon + 1)

        # Choose ECNN cell
        if lstm:
            self.ecnn_cell = ecnn_lstm_cell.ECNN_LSTM_Cell
        else:
            self.ecnn_cell = ecnn_cell.ECNNCell
        # Init ECNN cell
        self.ecnn_cell = self.ecnn_cell(
            n_features_U, n_state_neurons, activation, n_features_Y
        )

        self.init_state = nn.Parameter(
            torch.rand(1, n_state_neurons), requires_grad=learn_init_state
        )
        if init_state is not None:
            self.init_state.data = init_state

    def forward(self, U: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        U: torch.Tensor
            A batch of input features sequences for the ECNN.
            U should have shape=(past_horizon, batchsize, n_features_U) or
            shape=(past_horizon+forecast_horizon, batchsize, n_features_U),
            depending on future_U.
        Y: torch.Tensor
            A batch of output sequences for the ECNN.
            Y should have shape=(past_horizon, batchsize, n_features_Y).

        Returns
        -------
        torch.Tensor
            Contains past_error, the forecasting errors along the past_horizon
            where Y is known, and forecast, the forecast along the
            forecast_horizon. Both can be used for backpropagation.
            shape=(past_horizon+forecast_horizon, batchsize, n_features_Y)
        """

        # LSTM: Keep entries of diagonal matrix between 0 and 1.
        if self.lstm:
            diag_entries = torch.clamp(self.ecnn_cell.D.weight.data, 0, 1)
            self.ecnn_cell.D.weight.data = torch.nn.Parameter(diag_entries)

        # Check sizes of input and output
        self.check_sizes(U, Y)
        batchsize = U.shape[1]

        past_error = torch.empty(size=(self.past_horizon, batchsize, self.n_features_Y))
        forecast = torch.empty(
            size=(self.forecast_horizon, batchsize, self.n_features_Y)
        )

        approach = self.approach

        if approach == "backward":
            # start
            _, self.state[0] = self.ecnn_cell(
                self.init_state.repeat(batchsize, 1), U[0]
            )
            # past
            for t in range(1, self.past_horizon):
                past_error[t - 1], self.state[t] = self.ecnn_cell(
                    self.state[t - 1], U[t], Y[t - 1]
                )
            if self.future_U:
                past_error[t], self.state[t + 1] = self.ecnn_cell(
                    self.state[t], U=U[t + 1], Y=Y[t]
                )
                # future
                for t in range(
                    self.past_horizon + 1, self.past_horizon + self.forecast_horizon
                ):
                    forecast[t - self.past_horizon - 1], self.state[t] = self.ecnn_cell(
                        self.state[t - 1], U=U[t]
                    )
                forecast[t - self.past_horizon], self.state[t + 1] = self.ecnn_cell(
                    self.state[t]
                )
            else:
                past_error[t], self.state[t + 1] = self.ecnn_cell(self.state[t], Y=Y[t])
                # future
                for t in range(
                    self.past_horizon + 1, self.past_horizon + self.forecast_horizon + 1
                ):
                    forecast[t - self.past_horizon - 1], self.state[t] = self.ecnn_cell(
                        self.state[t - 1]
                    )

        if approach == "forward":
            # start
            self.state[0] = self.init_state.repeat(batchsize, 1)
            # past
            for t in range(1, self.past_horizon + 1):
                past_error[t - 1], self.state[t] = self.ecnn_cell(
                    self.state[t - 1], U[t - 1], Y[t - 1]
                )
            # future
            if self.future_U:
                for t in range(
                    self.past_horizon + 1, self.past_horizon + self.forecast_horizon + 1
                ):
                    forecast[t - self.past_horizon - 1], self.state[t] = self.ecnn_cell(
                        self.state[t - 1], U=U[t - 1]
                    )
            else:
                for t in range(
                    self.past_horizon + 1, self.past_horizon + self.forecast_horizon + 1
                ):
                    forecast[t - self.past_horizon - 1], self.state[t] = self.ecnn_cell(
                        self.state[t - 1]
                    )

        return torch.cat((past_error, forecast), dim=0)

    def check_sizes(self, U: torch.Tensor, Y: torch.Tensor) -> None:
        """Checks if U and Y have right shape."""

        if self.future_U:
            if (
                U.size()[0] != self.past_horizon + self.forecast_horizon
                or U.size()[2] != self.n_features_U
            ):
                raise ValueError(
                    "U must be of shape=(past_horizon+forecast_horizon, batchsize, n_features_U)."
                    " Have you initialized ECNN with the"
                    " right parameters?"
                )
        else:
            if U.size()[0] != self.past_horizon or U.size()[2] != self.n_features_U:
                raise ValueError(
                    "U must be of shape=(past_horizon, batchsize, n_features_U)."
                    " Have you initialized ECNN with the"
                    " right parameters?"
                )
        if Y.size()[0] != self.past_horizon or Y.size()[2] != self.n_features_Y:
            raise ValueError(
                "Y must be of shape=(past_horizon, batchsize, n_features_Y)."
                " Have you initialized ECNN with the"
                " right parameters?"
            )
        if Y.size()[1] != U.size()[1]:
            raise ValueError(
                "U and Y must have the same batchsize, but dim 1 of U is unequal dim 1 of Y."
            )

    def _check_variables(self) -> None:
        """
        Checks if self.n_state_neurons, self.n_features_Y, self.n_features_U, self.past_horizon,
        self.forecast_horizon, self.approach, self.future_U have valid inputs.

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
        if (self.n_features_U < 1) or (type(self.n_features_U) != int):
            raise ValueError(
                "{} is not a valid number for n_features_U. "
                "It must be an integer greater than 0.".format(self.n_features_U)
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
        if (self.approach not in ["backward", "forward"]) or (
            type(self.approach) != str
        ):
            raise ValueError(
                "{} is not a valid string for approach. "
                'It must be either "backward" or "forward".'.format(self.approach)
            )
        if (self.approach == "forward") and (self.past_horizon <= 1):
            raise ValueError(
                "The past_horizon must be larger than 1, "
                "otherwise B will not be trained."
            )
        if type(self.future_U) != bool:
            raise ValueError(
                "{} is not a valid input for approach. "
                "It must be boolean.".format(self.future_U)
            )
