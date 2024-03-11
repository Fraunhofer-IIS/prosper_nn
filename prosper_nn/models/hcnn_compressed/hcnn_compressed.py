import torch.nn as nn
import torch
from typing import Optional, Type
from . import hcnn_cell_compressed


class HCNN_compressed(nn.Module):
    """
    The HCNN_compressed class creates a Historical Consistent Neural Network that makes a difference
    between the features you are really interested in (task) and the features you use to improve
    the forecast of the task variables (support).
    It is possible to compress the support variables before they are compared with the expectation
    in the model. In that way, the number of features in the target layers get decreased.
    """

    def __init__(
        self,
        n_state_neurons: int,
        n_features_task: int,
        n_features_sup: int,
        n_features_sup_comp: int,
        past_horizon: int,
        forecast_horizon: int,
        sparsity: float = 0.0,
        activation: Type[torch.autograd.Function] = torch.tanh,
        init_state: Optional[torch.Tensor] = None,
        learn_init_state: bool = True,
        teacher_forcing: float = 1,
        decrease_teacher_forcing: float = 0,
    ):
        """
        Parameters
        ----------
        n_state_neurons : int
            The dimension of the state in the HCNN Cell. It must be a positive integer.
        n_features_task : int
            The size of the task variables to predict in each timestamp.
            It must be a positive integer.
        n_features_sup: int
            The size of the support variables which are input in each timestamp.
            It must be a positive integer.
        n_features_sup_comp: int
            The size to which we are compressing our support variables in each timestamp.
            It must be a positive integer.
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
        super(HCNN_compressed, self).__init__()
        self.n_state_neurons = n_state_neurons
        self.n_features_task = n_features_task
        self.n_features_sup = n_features_sup
        self.n_features_sup_comp = n_features_sup_comp
        self.past_horizon = past_horizon
        self.forecast_horizon = forecast_horizon
        self.sparsity = sparsity
        self.activation = activation
        self.teacher_forcing = teacher_forcing
        self.decrease_teacher_forcing = decrease_teacher_forcing
        self.state = [torch.tensor for _ in range(past_horizon + forecast_horizon + 1)]

        self._check_variables()

        self.init_state = nn.Parameter(
            torch.randn(1, n_state_neurons), requires_grad=learn_init_state
        )
        if init_state is not None:
            self.init_state.data = init_state

        self.HCNNCell = hcnn_cell_compressed.HCNNCell_compressed(
            self.n_state_neurons,
            self.n_features_task,
            self.n_features_sup,
            self.n_features_sup_comp,
            self.sparsity,
            self.activation,
            self.teacher_forcing,
        )

    def forward(self, task: torch.Tensor, support: torch.Tensor):
        """
        Parameters
        ----------
        task : torch.Tensor
            task should be 3-dimensional with the shape = (past_horizon, batchsize, n_features_task).
            This timeseries of observations contains the features we want to predict in future.
            Used for training the model.
        support : torch.Tensor
            support should be 3-dimensional with the shape = (past_horizon, batchsize, n_features_sup).
            This timeseries of observations contains the features that influence the task we want to predict.
            predict in future.
            Used for training the model.

        Returns
        -------
        torch.Tensor
            Contains past_error, the forecasting errors along the past_horizon
            where Y is known, and forecast, the forecast along the
            forecast_horizon. Both can be used for backpropagation.
            shape=(past_horizon+forecast_horizon, batchsize, n_features_Y)
        """

        self.state[0] = self.init_state

        self._check_sizes(task, support)

        batchsize = task.shape[1]

        # reset saved cell outputs
        past_error_task = torch.zeros(
            self.past_horizon, batchsize, self.n_features_task
        )

        past_error_support = torch.zeros(
            self.past_horizon, batchsize, self.n_features_sup_comp
        )

        forecast_task = torch.zeros(
            self.forecast_horizon, batchsize, self.n_features_task
        )

        forecast_support = torch.zeros(
            self.forecast_horizon, batchsize, self.n_features_sup_comp
        )

        # past
        for t in range(self.past_horizon):
            if t == 0:
                (
                    self.state[t + 1],
                    past_error_task[t],
                    past_error_support[t],
                ) = self.HCNNCell(
                    self.state[t].repeat(batchsize, 1), task[t], support[t]
                )
            else:
                (
                    self.state[t + 1],
                    past_error_task[t],
                    past_error_support[t],
                ) = self.HCNNCell(self.state[t], task[t], support[t])
        # future
        for t in range(self.past_horizon, self.past_horizon + self.forecast_horizon):
            (
                self.state[t + 1],
                forecast_task[t - self.past_horizon],
                forecast_support[t - self.past_horizon],
            ) = self.HCNNCell(self.state[t])

        return torch.cat([past_error_task, forecast_task], dim=0), torch.cat(
            [past_error_support, forecast_support], dim=0
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
            self.HCNNCell.set_teacher_forcing(new_teacher_forcing)

    def _check_sizes(self, task: torch.Tensor, support: torch.Tensor) -> None:
        """
        Checks if task and support has right dimensions.
        Parameters
        ----------
        task : torch.Tensor
            task should be 3-dimensional with the shape = (past_horizon, batchsize, n_features_task).
            This timeseries of observations contains the features we want to predict in future.

        support : torch.Tensor
            support should be 3-dimensional with the shape = (past_horizon, batchsize, n_features_sup).
            This timeseries of observations contains the features that inffluence the task that want to
            predict in future.

        Returns
        -------
        None
        """

        if len(task.shape) != 3:
            raise ValueError(
                "The shape for a batch of observations in task should be "
                "shape = (past_horizon, batchsize, n_features_task)"
            )

        if (task.shape[0] != self.past_horizon) or (
            task.shape[2] != self.n_features_task
        ):
            raise ValueError(
                "task must be of the dimensions"
                " shape = (past_horizon, batchsize, n_features_task)."
                " Have you initialized HCNN Compressed with the"
                " right parameters?"
            )

        # For support
        if len(support.shape) != 3:
            raise ValueError(
                "The shape for a batch of observations in support should be "
                "shape = (past_horizon, batchsize, n_features_sup)"
            )

        if (support.shape[0] != self.past_horizon) or (
            support.shape[2] != self.n_features_sup
        ):
            raise ValueError(
                "support must be of the dimensions"
                " shape = (past_horizon, batchsize, n_features_sup)."
                " Have you initialized HCNN compressed with the"
                " right parameters?"
            )

        if task.shape[1] != support.shape[1]:
            raise ValueError(
                "task and support should have same batchsize. "
                "Please check the dimensions with which the inputs"
                "task and support are initialized."
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

        if (self.n_features_task < 1) or (type(self.n_features_task) != int):
            raise ValueError(
                "{} is not a valid number for n_features_task. "
                "It must be an integer greater than 0.".format(self.n_features_task)
            )
        if self.n_features_task > self.n_state_neurons:
            raise ValueError(
                "{} is not a valid number for n_state_neurons. "
                "It must be greater than or equal to n_features_task ({})."
                "".format(self.n_state_neurons, self.n_features_task)
            )

        if (self.n_features_sup < 1) or (type(self.n_features_sup) != int):
            raise ValueError(
                "{} is not a valid number for n_features_sup. "
                "It must be an integer greater than 0.".format(self.n_features_sup)
            )
        if self.n_features_sup > self.n_state_neurons:
            raise ValueError(
                "{} is not a valid number for n_state_neurons. "
                "It must be greater than or equal to n_features_sup ({})."
                "".format(self.n_state_neurons, self.n_features_sup)
            )

        if (self.n_features_sup_comp < 1) or (type(self.n_features_sup_comp) != int):
            raise ValueError(
                "{} is not a valid number for n_features_sup_comp. "
                "It must be an integer greater than 0.".format(self.n_features_sup_comp)
            )
        if self.n_features_sup_comp > self.n_state_neurons:
            raise ValueError(
                "{} is not a valid number for n_state_neurons. "
                "It must be greater than or equal to n_features_sup_comp ({})."
                "".format(self.n_state_neurons, self.n_features_task)
            )
        if self.n_features_sup_comp > self.n_features_sup:
            raise ValueError(
                "{} is not a valid number for n_features_sup_comp. "
                "It must be less than or equal to n_features_sup ({})."
                "".format(self.n_features_sup_comp, self.n_features_sup)
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
