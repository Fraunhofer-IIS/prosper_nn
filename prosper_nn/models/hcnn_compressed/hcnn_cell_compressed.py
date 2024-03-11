import torch.nn as nn
import torch
import torch.nn.utils.prune as prune
from typing import Optional, Type
from operator import xor


class HCNNCell_compressed(nn.Module):
    """
    The HCNNCell call is implemented to model one forecast step in a Historical Consistent Neural Network
    with compressed support input.
    By recursively using the cell a HCNN network can be implemented.
    """

    def __init__(
        self,
        n_state_neurons: int,
        n_features_task: int,
        n_features_sup: int,
        n_features_sup_comp: int,
        sparsity: float = 0.0,
        activation: Type[torch.autograd.Function] = torch.tanh,
        teacher_forcing: float = 1,
    ):
        """
        Parameters
        ----------
        n_state_neurons : int
            The dimension of the state in the HCNN Cell. It must be a positive integer.
        n_features_task : int
            The size of the task variables to predict in each timestamp.
            It must be a positive integer.
        n_festures_support: int
            The size of the support variables which are input in each timestamp.
            It must be a positive integer.
        n_features_compressed_support: int
            The size to which we are compressing our support variables in each timestamp.
            It must be a positive integer.
        sparsity : float
            The share of weights that are set to zero in the matrix A.
            These weights are not trainable and therefore always zero.
            For big matrices (dimension > 50) this can be necessary to guarantee
            numerical stability
            and increases the long-term memory of the model.
        activation : Type[torch.autograd.Function]
            The activation function that is applied on the output of the hidden layers.
            The same function is used on all hidden layers.
            No function is applied if no function is given.
        teacher_forcing : float
            The probability that teacher forcing is applied for a single state neuron.
            In each time step this is repeated and therefore enforces stochastic learning
            if the value is smaller than 1.

        Returns
        -------
        None
        """
        super(HCNNCell_compressed, self).__init__()
        self.n_state_neurons = n_state_neurons
        self.n_features_task = n_features_task
        self.n_features_sup = n_features_sup
        self.n_features_sup_comp = n_features_sup_comp
        self.sparsity = sparsity
        self.activation = activation
        self.teacher_forcing = teacher_forcing

        if type(activation) == str and activation == "torch.tanh":
            self.activation = torch.tanh

        self.A = nn.Linear(
            in_features=self.n_state_neurons,
            out_features=self.n_state_neurons,
            bias=False,
        )
        self.E = nn.Linear(
            in_features=self.n_features_sup,
            out_features=self.n_features_sup_comp,
            bias=False,
        )

        self.eye_task = nn.Parameter(torch.eye(
            self.n_features_task,
            self.n_state_neurons,
        ),requires_grad=False)

        self.eye_support = nn.Parameter(torch.cat(
            (
                torch.zeros(
                    self.n_features_sup_comp,
                    (self.n_state_neurons - self.n_features_sup_comp),
                ),
                torch.eye(self.n_features_sup_comp, self.n_features_sup_comp),
            ),
            1,
        ))

        self.ptf_dropout = nn.Dropout(1 - self.teacher_forcing)

        if self.sparsity > 0:
            prune.random_unstructured(self.A, name="weight", amount=self.sparsity)


    def forward(
        self,
        state: torch.Tensor,
        observation_task: Optional[torch.Tensor] = None,
        observation_support: Optional[torch.Tensor] = None,
    ):
        """
        Parameters
        ----------
        state : torch.Tensor
            The previous state of the HCNN. shape = (batch_size, n_state_neurons)
        observation_task : torch.Tensor
            The observation_task is the data for the given timestamp which should be predicted from supports.
            It has the
            shape = (batchsize, n_features_task).
            It is an optional variable. If no variable is given,
            the observation is not subtracted
            from the expectation to create the output variable.
            Additionally, no teacher forcing is applied on the state vector.
        observation_support : torch.Tensor
            The observation_support is the data for the given timestamp which is compressed and then used to learn observation_task.
            It has the
            shape = (batchsize, n_features_sup).
            It is an optional variable. If no variable is given,
            the observation is not subtracted
            from the expectation to create the output variable.
            Additionally, no teacher forcing is applied on the state vector.


        Returns
        -------
        state : torch.Tensor
            The updated state of the HCNN.
        output_task: torch.Tensor
            The output of the HCNN Cell. If a observation_task is given,
            this output is calculated by the expectation_task minus the observation_task.
            If no observation_task is given, the output is equal to the expectation.
        """

        expectation_task = torch.mm(state, self.eye_task.T)
        expectation_support = torch.mm(state, self.eye_support.T)

        if observation_task is not None and observation_support is not None:
            support_compressed = self.E(observation_support)

            output_task = expectation_task - observation_task
            output_support = expectation_support - support_compressed

            teacher_forcing_task = torch.mm(
                self.ptf_dropout(output_task), self.eye_task
            )
            teacher_forcing_support = torch.mm(
                self.ptf_dropout(output_support), self.eye_support
            )

            state = self.activation(
                state - teacher_forcing_task - teacher_forcing_support
            )

        elif xor(observation_task is None, observation_support is None):  # XOR only one of them is set
            self.set_task_and_support_error(observation_task, observation_support)

        else:  # Forecasts
            output_task = expectation_task
            output_support = expectation_support
            state = self.activation(state)
        state = self.A(state)
        return state, output_task, output_support

    def set_teacher_forcing(self, teacher_forcing: float) -> None:
        """
        Function to set teacher forcing to a specific value in layer and as self variable.

        Parameters
        ----------
        teacher_forcing: float
            The value teacher forcing is set to in the cell.

        Returns
        -------
        None
        """
        if (teacher_forcing < 0) or (teacher_forcing > 1):
            raise ValueError(
                "{} is not a valid number for teacher_forcing. "
                "It must be a value in the interval [0, 1].".format(teacher_forcing)
            )
        self.teacher_forcing = teacher_forcing
        self.ptf_dropout.p = 1 - teacher_forcing

    def set_task_and_support_error(self, observation_task, observation_support) -> None:
        """
        The observation_task and observation_support tensors should either both be set or both be not set.
        This is used to check and throw the error if either of them is empty and reminds to set that.


        Parameters
        ----------
        observation_task, observation_support

        Returns
        -------
        None
        """
        if observation_task is None:
            raise ValueError("observation_task is empty and please set it")
        elif observation_support is None:
            raise ValueError("observation_support is empty and please set it")
