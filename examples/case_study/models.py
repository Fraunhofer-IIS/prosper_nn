from typing import Union, Tuple

import torch
import torch.nn as nn
from prosper_nn.models.ecnn import ECNN
from prosper_nn.models.hcnn import HCNN
from prosper_nn.models.crcnn import CRCNN
from prosper_nn.models.ensemble import Ensemble
from fuzzy_nn import Benchmark_Fuzzy_NN


class Benchmark_RNN(nn.Module):
    """
    Parent class to create various RNNs based on Elman, GRU and LSTM cells.
    Additionally, the forecast methods direct, recursive and
    sequence to sequence (S2S) are possible.
    For all approaches the past_target is merged to the features_past to enable
    an autoregressive part in the models.
    """

    def __init__(
        self,
        n_features_U: int,
        n_state_neurons: int,
        n_features_Y: int,
        forecast_horizon: int,
        cell_type: str,
    ):
        super(Benchmark_RNN, self).__init__()
        self.multivariate = False

        self.n_features_Y = n_features_Y
        self.forecast_horizon = forecast_horizon
        self.n_state_neurons = n_state_neurons
        self.cell_type = cell_type

        self.cell = self.get_recurrent_cell()
        self.rnn = self.cell(input_size=n_features_U, hidden_size=n_state_neurons)
        self.state_output = nn.Linear(
            in_features=n_state_neurons, out_features=self.output_size_linear_decoder
        )
        self.init_state = self.set_init_state()

    def forward(self, features_past: torch.Tensor) -> torch.Tensor:
        batchsize = features_past.size(1)

        init_state = self.repeat_init_state(batchsize)
        output_rnn = self.rnn(features_past, init_state)
        return output_rnn

    def set_init_state(self) -> Union[nn.Parameter, Tuple[nn.Parameter, nn.Parameter]]:
        dtype = torch.float64
        if self.cell_type == "lstm":
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
        if self.cell_type == "elman":
            cell = nn.RNN
        elif self.cell_type == "gru":
            cell = nn.GRU
        elif self.cell_type == "lstm":
            cell = nn.LSTM
        else:
            raise ValueError(
                f"cell_type {self.cell_type} not available."
            )
        return cell

    def repeat_init_state(
        self, batchsize: int
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if self.cell_type == "lstm":
            return self.init_state[0].repeat(batchsize, 1).unsqueeze(
                0
            ), self.init_state[1].repeat(batchsize, 1).unsqueeze(0)
        else:
            return self.init_state.repeat(batchsize, 1).unsqueeze(0)

    def get_input(self, features_past, target_past):
        return (features_past,)

    def extract_forecasts(self, ensemble_output):
        mean = ensemble_output[-1]

        return mean


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
        cell_type: str,
    ):
        self.forecast_method = "direct"
        self.output_size_linear_decoder = n_features_Y * forecast_horizon
        super(RNN_direct, self).__init__(
            n_features_U,
            n_state_neurons,
            n_features_Y,
            forecast_horizon,
            cell_type,
        )

    def forward(self, features_past: torch.Tensor) -> torch.Tensor:
        output_rnn = super(RNN_direct, self).forward(features_past)

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
        cell_type: str,
    ):
        self.forecast_method = "recursive"
        self.output_size_linear_decoder = n_features_Y
        super(RNN_recursive, self).__init__(
            n_features_U,
            n_state_neurons,
            n_features_Y,
            forecast_horizon,
            cell_type,
        )

    def forward(self, features_past: torch.Tensor) -> torch.Tensor:
        # add zeros as RNN input for forecast horizon
        future_zeros_features = torch.zeros_like(features_past)[: self.forecast_horizon]
        features = torch.cat([features_past, future_zeros_features], dim=0)

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
        cell_type: str,
    ):
        self.forecast_method = "s2s"
        self.output_size_linear_decoder = n_features_Y
        super(RNN_S2S, self).__init__(
            n_features_U,
            n_state_neurons,
            n_features_Y,
            forecast_horizon,
            cell_type,
        )
        self.decoder = self.cell(input_size=n_features_U, hidden_size=n_state_neurons)

    def forward(self, features_past: torch.Tensor) -> torch.Tensor:
        output_rnn = super(RNN_S2S, self).forward(features_past)

        # add dummy zeros as RNN input for forecast horizon
        future_zeros_features = torch.zeros_like(features_past)[: self.forecast_horizon]

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

    multivariate = False

    def __init__(
        self, past_horizon: int, forecast_horizon: int, n_features_Y: int
    ) -> None:
        super().__init__()
        self.past_horizon = past_horizon
        self.forecast_horizon = forecast_horizon
        self.n_features_Y = n_features_Y

    def forward(self, features_past: torch.Tensor) -> torch.Tensor:
        return torch.zeros(
            self.forecast_horizon,
            features_past.size(1),
            self.n_features_Y,
        )

    def get_input(self, features_past, target_past):
        return (features_past,)

    def extract_forecasts(self, output):
        return output


class Benchmark_ECNN(ECNN):
    """
    Adds methods to run ECNN in unified benchmark pipeline.
    """

    multivariate = False

    def get_input(self, features_past, target_past):
        return features_past, target_past

    def extract_forecasts(self, ensemble_output):
        mean = ensemble_output[-1]
        _, forecasts = torch.split(mean, self.past_horizon)
        return forecasts


class Benchmark_HCNN(HCNN):
    """
    Adds methods to run HCNN in unified benchmark pipeline.
    """

    multivariate = True

    def get_input(self, features_past, target_past):
        return (features_past,)

    def extract_forecasts(self, ensemble_output):
        mean = ensemble_output[-1]
        _, forecasts = torch.split(mean, self.past_horizon)
        return forecasts


class Benchmark_CRCNN(CRCNN):
    """
    Adds methods to run CRCNN in unified benchmark pipeline.
    """

    multivariate = True

    def get_input(self, features_past, target_past):
        return (features_past,)

    def extract_forecasts(self, ensemble_output):
        mean = ensemble_output[-1, -1]
        _, forecasts = torch.split(mean, self.past_horizon)
        return forecasts


def init_models(
    benchmark_models,
    n_features_U,
    n_state_neurons,
    n_features_Y,
    past_horizon,
    forecast_horizon,
    n_models,
):
    # Error Correction Neural Network (ECNN)
    ecnn = Benchmark_ECNN(
        n_state_neurons=n_state_neurons,
        n_features_U=n_features_U,
        n_features_Y=n_features_Y,
        past_horizon=past_horizon,
        forecast_horizon=forecast_horizon,
    ).double()

    # Define an Ensemble for better forecasts, heatmap visualization and sensitivity analysis
    ecnn_ensemble = init_ensemble(ecnn, n_models)

    benchmark_models["ECNN"] = ecnn_ensemble

    # # Fuzzy Neural Network
    # fuzzy_nn = Benchmark_Fuzzy_NN(n_features_input=n_features_U, past_horizon=past_horizon,)
    # fuzzy_nn_ensemble = init_ensemble(fuzzy_nn, n_models, keep_pruning_mask=True)
    # benchmark_models["Fuzzy_NN"] = fuzzy_nn_ensemble

    if not 'HCNN' in benchmark_models: # Reuse trained multivariate model
        # Historical Consistent Neural Network (HCNN)
        hcnn = Benchmark_HCNN(
            n_state_neurons=n_state_neurons,
            n_features_Y=n_features_U,
            past_horizon=past_horizon,
            forecast_horizon=forecast_horizon,
        )
        hcnn_ensemble = init_ensemble(hcnn, n_models)
        benchmark_models["HCNN"] = hcnn_ensemble

    if not 'CRCNN' in benchmark_models: # Reuse trained multivariate model
        # Causal Retro Causal Neural Network (CRCNN)
        crcnn = Benchmark_CRCNN(
            n_state_neurons=n_state_neurons,
            n_features_Y=n_features_U,
            past_horizon=past_horizon,
            forecast_horizon=forecast_horizon,
        )
        crcnn_ensemble = init_ensemble(crcnn, n_models)
        benchmark_models["CRCNN"] = crcnn_ensemble

    # Compare to further Recurrent Neural Networks
    for forecast_module in [RNN_direct, RNN_recursive, RNN_S2S]:
        for cell_type in ["elman", "gru", "lstm"]:
            model = forecast_module(
                n_features_U,
                n_state_neurons,
                n_features_Y,
                forecast_horizon,
                cell_type,
            )
            ensemble = init_ensemble(model, n_models)
            benchmark_models[f"{cell_type}_{model.forecast_method}"] = (
                ensemble
            )

    # Additionally, compare with the naive no-change forecast
    benchmark_models["Naive"] = Naive(past_horizon, forecast_horizon, n_features_Y)

    return benchmark_models


def init_ensemble(model, n_models, keep_pruning_mask=False):
    ensemble = Ensemble(
        model=model, n_models=n_models, keep_pruning_mask=keep_pruning_mask
    ).double()
    setattr(ensemble, "get_input", model.get_input)
    setattr(ensemble, "extract_forecasts", model.extract_forecasts)
    setattr(ensemble, "multivariate", model.multivariate)
    return ensemble
