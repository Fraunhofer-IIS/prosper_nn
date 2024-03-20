from typing import Union, Tuple

import torch
import torch.nn as nn


class Benchmark_RNN(nn.Module):
    """
    Parent class to create various RNNs based on Elman, GRU and LSTM cells.
    Additionally, the forecast methods direct, recursive and
    sequence to sequence (S2S) are possible.
    For all approaches the past_target is merged to the past_features to enable
    an autoregressive part in the models.
    """

    def __init__(
        self,
        n_features_U: int,
        n_state_neurons: int,
        n_features_Y: int,
        forecast_horizon: int,
        recurrent_cell_type: str,
    ):
        super(Benchmark_RNN, self).__init__()
        self.n_features_Y = n_features_Y
        self.forecast_horizon = forecast_horizon
        self.n_state_neurons = n_state_neurons
        self.recurrent_cell_type = recurrent_cell_type

        self.cell = self.get_recurrent_cell()
        self.rnn = self.cell(input_size=n_features_U, hidden_size=n_state_neurons)
        self.state_output = nn.Linear(
            in_features=n_state_neurons, out_features=self.output_size_linear_decoder
        )
        self.init_state = self.set_init_state()

    def forward(self, past_features: torch.Tensor) -> torch.Tensor:
        batchsize = past_features.size(1)

        init_state = self.repeat_init_state(batchsize)
        output_rnn = self.rnn(past_features, init_state)
        return output_rnn

    def set_init_state(self) -> Union[nn.Parameter, Tuple[nn.Parameter, nn.Parameter]]:
        dtype = torch.float64
        if self.recurrent_cell_type == "lstm":
            init_state = (
                nn.Parameter(
                    torch.rand(1, self.n_state_neurons, dtype=dtype), requires_grad=True
                ),
                nn.Parameter(
                    torch.rand(1, self.n_state_neurons, dtype=dtype), requires_grad=True
                ),
            )
        else:
            init_state = nn.Parameter(
                torch.rand(1, self.n_state_neurons, dtype=dtype), requires_grad=True
            )
        return init_state

    def get_recurrent_cell(self) -> nn.Module:
        if self.recurrent_cell_type == "elman":
            cell = nn.RNN
        elif self.recurrent_cell_type == "gru":
            cell = nn.GRU
        elif self.recurrent_cell_type == "lstm":
            cell = nn.LSTM
        else:
            raise ValueError(
                f"recurrent_cell_type {self.recurrent_cell_type} not available."
            )
        return cell

    def repeat_init_state(
        self, batchsize: int
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if self.recurrent_cell_type == "lstm":
            return self.init_state[0].repeat(batchsize, 1).unsqueeze(
                0
            ), self.init_state[1].repeat(batchsize, 1).unsqueeze(0)
        else:
            return self.init_state.repeat(batchsize, 1).unsqueeze(0)


class RNN_direct(Benchmark_RNN):
    """
    Encodes the data of the past horizon into a context vector with a RNN.
    Afterwards, the context vector is mapped to the forecasts of all
    forecast steps in the forecast horizon by an affine linear transformation.

    .. math::

        s_0, \dots, s_T = rnn(y_0, \dots, y_T; s_0)

        \hat{y}_{T+1}, \dots, \hat{y}_{T+\tau} = A \cdot s_{T}
        with A \in \mathbb{R}^{n_features_Y \cdot forecast_horizon \times n_state_neurons}
    """

    def __init__(
        self,
        n_features_U: int,
        n_state_neurons: int,
        n_features_Y: int,
        forecast_horizon: int,
        recurrent_cell_type: str,
    ):
        self.forecast_method = "direct"
        self.output_size_linear_decoder = n_features_Y * forecast_horizon
        super(RNN_direct, self).__init__(
            n_features_U,
            n_state_neurons,
            n_features_Y,
            forecast_horizon,
            recurrent_cell_type,
        )

    def forward(self, past_features: torch.Tensor) -> torch.Tensor:
        output_rnn = super(RNN_direct, self).forward(past_features)

        context_vector = output_rnn[0][-1]
        forecast = self.state_output(context_vector)
        forecast = forecast.reshape(self.forecast_horizon, -1, self.n_features_Y)
        return forecast


class RNN_recursive(Benchmark_RNN):
    """
    Encodes the data of the past horizon into a context vector with a RNN.
    The context vector is given to the same RNN to predict states for the forecast horizon.
    Each state in the forecast horizon is then decoded by an affine linear
    transformation A.

    .. math::

        s_0, \dots, s_T = rnn(y_0, \dots, y_T; s_0)
        s_{T+1}, ..., s_{T+\tau} = rnn(0_{T+1}, \dots, 0_{T+\tau}; s_T)

        \hat{y}_{T+i} = A \cdot s_{T+i} for i=1,...,\tau
        with A \in \mathbb{R}^{n_features_Y \times n_state_neurons}

    """

    def __init__(
        self,
        n_features_U: int,
        n_state_neurons: int,
        n_features_Y: int,
        forecast_horizon: int,
        recurrent_cell_type: str,
    ):
        self.forecast_method = "recursive"
        self.output_size_linear_decoder = n_features_Y
        super(RNN_recursive, self).__init__(
            n_features_U,
            n_state_neurons,
            n_features_Y,
            forecast_horizon,
            recurrent_cell_type,
        )

    def forward(self, past_features: torch.Tensor) -> torch.Tensor:
        # add zeros as RNN input for forecast horizon
        future_zeros_features = torch.zeros_like(past_features)[: self.forecast_horizon]
        features = torch.cat([past_features, future_zeros_features], dim=0)

        output_rnn = super(RNN_recursive, self).forward(features)
        future_states = output_rnn[0][-self.forecast_horizon :]
        forecast = self.state_output(future_states)
        return forecast


class RNN_S2S(Benchmark_RNN):
    """
    Encodes the data of the past horizon into a context vector with a RNN.
    The context vector is given to another RNN of the same recurrent cell type
    to predict a state for each step in the forecast horizon.
    Each state in the forecast horizon is then decoded by an affine linear
    transformation A.

    .. math::

        s_0, \dots, s_T = rnn(y_0, \dots, y_T; s_0)
        s_{T+1}, ..., s_{T+\tau} = \tilde{rnn}(0_{T+1}, \dots, 0_{T+\tau}; s_T)

        \hat{y}_{T+i} = A \cdot s_{T+i} for i=1,...,\tau
        with A \in \mathbb{R}^{n_features_Y \times n_state_neurons}
    """

    def __init__(
        self,
        n_features_U: int,
        n_state_neurons: int,
        n_features_Y: int,
        forecast_horizon: int,
        recurrent_cell_type: str,
    ):
        self.forecast_method = "s2s"
        self.output_size_linear_decoder = n_features_Y
        super(RNN_S2S, self).__init__(
            n_features_U,
            n_state_neurons,
            n_features_Y,
            forecast_horizon,
            recurrent_cell_type,
        )
        self.decoder = self.cell(input_size=n_features_U, hidden_size=n_state_neurons)

    def forward(self, past_features: torch.Tensor) -> torch.Tensor:
        output_rnn = super(RNN_S2S, self).forward(past_features)

        # add dummy zeros as RNN input for forecast horizon
        future_zeros_features = torch.zeros_like(past_features)[: self.forecast_horizon]

        context_vector = output_rnn[1]
        states_decoder = self.decoder(future_zeros_features, context_vector)[0]
        forecast = self.state_output(states_decoder)
        return forecast


class Naive(nn.Module):
    """
    Model that predicts zero changes. Implemented so that it can be used like the
    other benchmark recurrent neural networks.

    .. math::

        \hat{y}_{T+i} = 0 for i=1,...,\tau
    """

    def __init__(
        self, past_horizon: int, forecast_horizon: int, n_features_Y: int
    ) -> None:
        super().__init__()
        self.past_horizon = past_horizon
        self.forecast_horizon = forecast_horizon
        self.n_features_Y = n_features_Y

    def forward(self, past_features: torch.Tensor) -> torch.Tensor:
        return torch.zeros(
            self.forecast_horizon,
            past_features.size(1),
            self.n_features_Y,
        )
