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
import torch.nn.utils.prune as prune
from typing import Optional, Type
from ..hcnn.hcnn_cell import no_dropout_backward


class HCNN_KNOWN_U_Cell(nn.Module):
    """
    The HCNN_KNOWN_U_Cell call is implemented to model one forecast step in a
    Historical Consistent Neural Network where we use some features U known for the future.
    By recursively using the cell a HCNN_KNOWN_U network can be implemented.
    Mathematically the the output of one cell is calculated as:

    .. math ::
        s_{t+1} = A \\tanh \\left( [\\mathbb{1}, 0]^T \cdot s_t -[\\mathbb{1}, 0, 0]^T \\cdot ( [\\mathbb{1}, 0] s_t -y_t^d) + [0, 0, \\mathbb{1}]^T u_t \\right)
    .. math ::
        y_t = [\\mathbb{1}, 0] \\cdot s_t

    """

    def __init__(
        self,
        n_state_neurons: int,
        n_features_U: int,
        n_features_Y: int,
        sparsity: float = 0.0,
        activation: Type[torch.autograd.Function] = torch.tanh,
        teacher_forcing: float = 1,
        backward_full_Y: bool = True,
        ptf_in_backward: bool = True,
    ):
        """
        Parameters
        ----------
        n_state_neurons : int
            The dimension of the state in the HCNN_KNOWN_U Cell. Must be a positive integer
            with n_state_neurons >= n_features_Y.
        n_features_U : int
            The size of the features U known for the future in each timestamp. Must be a positive integer.
        n_features_Y : int
            The number of features (including targets) whose dynamics are supposed to be modeled
            (and forecasted) internally. Must be a positive integer.
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
        backward_full_Y: bool
            Apply partial teacher forcing dropout after or before the output is calculated.
            If True dropout layer is applied afterwards and the output contains the errors of all features.
            If False dropout is applied before and the output contains only the errors that are not dropped.
            The remaining entries are zero and therefore contain no error
        ptf_in_backward: bool
            If True nothing happens and the Dropout layer is handled as it is in the backward path.
            If False the Dropout layer is skipped in the backward path.

        Returns
        -------
        None
        """
        super(HCNN_KNOWN_U_Cell, self).__init__()
        self.n_state_neurons = n_state_neurons
        self.n_features_Y = n_features_Y
        self.n_features_U = n_features_U
        self.sparsity = sparsity
        self.activation = activation
        self.teacher_forcing = teacher_forcing
        self.backward_full_Y = backward_full_Y
        self.ptf_in_backward = ptf_in_backward

        if type(activation) == str and activation == "torch.tanh":
            self.activation = torch.tanh

        # from state to output, used in filtering the hidden neurons.
        self.eye = torch.eye(
            self.n_features_Y, self.n_state_neurons, requires_grad=False
        )

        # from error with padding for hidden neurons and n_features_U to r state
        self.eye_err2rstate = torch.eye(
            self.n_features_Y,
            self.n_state_neurons + self.n_features_U,
            requires_grad=False,
        )

        # from state with padding for n_features_U to r state
        self.eye_state2rstate = torch.eye(
            self.n_state_neurons,
            self.n_state_neurons + self.n_features_U,
            requires_grad=False,
        )

        # from n_features_U with padding for n_state_neurons to r state.
        padding = torch.zeros(
            self.n_features_U,
            self.n_state_neurons,
        )
        eye_u = torch.eye(self.n_features_U)
        self.eye_U2rstate = torch.cat((padding, eye_u), axis=1)

        # from r state to state[t+1]
        self.A = nn.Linear(
            in_features=self.n_state_neurons + self.n_features_U,
            out_features=self.n_state_neurons,
            bias=False,
        )

        self.ptf_dropout = nn.Dropout(1 - self.teacher_forcing)
        if self.sparsity > 0:
            prune.random_unstructured(self.A, name="weight", amount=self.sparsity)
        if not self.ptf_in_backward:
            self.ptf_dropout.register_full_backward_hook(no_dropout_backward)

    def forward(
        self,
        state: torch.Tensor,
        known_features: torch.Tensor = None,
        observation: Optional[torch.Tensor] = None,
    ):
        """
        Parameters
        ----------
        state : torch.Tensor
            The previous state of the HCNN. shape = (batch_size, n_state_neurons)
        observation : torch.Tensor
            The observation is the data for the given timestamp which should be learned.
            It contains all observational features in the batch and has the
            shape = (batchsize, n_features_Y).
            It is an optional variable. If no variable is given,
            the observation is not subtracted
            from the expectation to create the output variable.
            Additionally, no teacher forcing is applied on the state vector.
        known_features : torch.Tensor
            The input for the known features at time t.
            known_features should have shape=(batchsize, n_features_U).

        Returns
        -------
        state : torch.Tensor
            The updated state of the HCNN_KNOWN_U.
        output: torch.Tensor
            The output of the HCNN_KNOWN_U_Cell. If an observation is given,
            this output is calculated by the expectation minus the observation.
            If no observation is given, the output is equal to the expectation.
        """

        # Cell forward calculations
        expectation = torch.mm(state, self.eye.T)
        if observation is not None:
            if self.backward_full_Y:
                output = expectation - observation
                teacher_forcing = torch.mm(
                    self.ptf_dropout(output), self.eye_err2rstate
                )
            elif not self.backward_full_Y:
                output = self.ptf_dropout(expectation - observation)
                teacher_forcing = torch.mm(output, self.eye_err2rstate)
            # state to rstate
            s2r = torch.mm(state, self.eye_state2rstate)
            # known to rstate
            k2r = torch.mm(known_features, self.eye_U2rstate)
            # the rstate
            rstate = s2r + k2r - teacher_forcing

            rstate = self.activation(rstate)

        else:  # Forecasts
            output = expectation
            # state to rstate
            s2r = torch.mm(state, self.eye_state2rstate)
            # known to rstate
            k2r = torch.mm(known_features, self.eye_U2rstate)
            # the rstate
            rstate = s2r + k2r

            rstate = self.activation(rstate)

        # for state + 1
        state = self.A(rstate)

        return state, output

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
