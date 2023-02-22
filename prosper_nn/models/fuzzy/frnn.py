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
from .fuzzification import Fuzzification
from .fuzzy_inference import FuzzyInference
from .defuzzification import Defuzzification
from typing import Any
import torch.nn.utils.prune as prune
import numpy as np


class FRNN(torch.nn.Module):
    """
    Fuzzy Recurrent Neural Network for classification. Wraps Fuzzy Layer in RNN.

    """

    def __init__(
        self,
        n_features_input: int,
        n_output_classes: int,
        n_rules: int,
        n_membership_fcts: int,
        membership_fcts: dict,
        rule_matrix: np.array,
        classification_matrix: Any = None,
        n_layers: int = 1,
        batch_first: bool = True,
        learn_conditions: bool = False,
        pruning: bool = True,
    ) -> None:
        """
        Parameters
        ----------
        n_features_input : int
            Number of network inputs
        n_output_classes : int
            Number of network outputs/classes
        n_rules : int
            Number of rules
        n_membership_fcts : int
            Number of membership functions
        membership_fcts : dict
            Dictionary containing the membership functions
        rule_matrix : np.array
            Array containing the rule weight matrix
        classification_matrix : np.array = None
            Array containing the matrix stating which rule leads to which class.
        n_layers : int = 1
            Number of recurrent layers in the RNN part of the network
        batch_first : bool = True
            Changes RNN to accommodate to a sequence or batch first data structure.
        learn_conditions : bool = False
            Set fuzzy inference learning mode
        pruning : bool = True
            Set fuzzy inference pruning mode
        """
        super(FRNN, self).__init__()

        # Defining parameters
        self.n_features_input = n_features_input
        self.n_output_classes = n_output_classes
        self.n_rules = n_rules
        self.n_membership_fcts = n_membership_fcts
        self.membership_fcts = membership_fcts
        self.rule_matrix = rule_matrix
        self.classification_matrix = classification_matrix
        self.n_layers = n_layers
        self.batch_first = batch_first
        self.learn_conditions = learn_conditions
        self.pruning = pruning

        # Defining the layers
        # input_size == hidden_size -> the hidden state should resemble the time dependence of
        # each input
        self.rnn = nn.RNN(
            input_size=n_features_input,
            hidden_size=n_features_input,
            num_layers=n_layers,
            batch_first=batch_first,
        )
        prune_identity = torch.eye(n_features_input)
        # Restrict RNN weights so the hidden state and the output resemble the time dependency of
        # the input sequence
        # weights are restricted to a diagonal matrix
        prune.custom_from_mask(self.rnn, "weight_ih_l0", prune_identity)
        prune.custom_from_mask(self.rnn, "weight_hh_l0", prune_identity)

        # Fuzzy layer
        self.fuzzy = nn.Sequential(
            Fuzzification(
                n_features_input=self.n_features_input, membership_fcts=self.membership_fcts
            ),
            FuzzyInference(
                n_features_input=self.n_features_input,
                n_membership_fctship_fcts=self.n_membership_fcts,
                n_rules=self.n_rules,
                n_output_classes=n_output_classes,
                rule_matrix=self.rule_matrix,
                learn_conditions=self.learn_conditions,
                prune_weights=self.pruning,
                classification_matrix=classification_matrix,
                learn_consequences=True,
            ),
        )

    def forward(self, inputs: torch.Tensor) -> None:
        """
        Network forward pass.

        Parameters
        ----------
        inputs : torch.Tensor
            Input tensor

        Returns
        -------
        output : torch.Tensor
            Output tensor
        """
        batchsize = inputs.size(0)
        # Initializing hidden_state state for first input using method defined below
        hidden_state = self.init_hidden(batchsize)

        # Passing in the input and hidden_state state into the model and obtaining outputs
        output, hidden_state = self.rnn(inputs, hidden_state)
        # get only output after the last time step
        output = output[:, -1]
        # pass through the fuzzy architecture
        output = self.fuzzy(output)
        return output

    def init_hidden(self, batchsize: int) -> torch.Tensor:
        """
        This method generates the first hidden_state state of zeros which is used in the forward pass.

        Parameters
        ----------
        batchsize : int
            number of Samples in one batch

        Returns
        -------
        hidden_state : torch.Tensor
            newly initialized hidden_state state

        """
        hidden_state = torch.zeros(
            self.n_layers, batchsize, self.n_features_input
        ).type(torch.DoubleTensor)
        return hidden_state
