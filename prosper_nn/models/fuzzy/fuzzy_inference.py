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
from copy import deepcopy
from typing import Any
import numpy as np
import torch.nn.utils.prune as prune


class FuzzyInference(torch.nn.Module):
    """
    The module performs the fuzzy inference step of a Fuzzy Neural Network.
    First, the conditions are modeled in a dense layer with rule constraints.
    Weights defined in the rule_matrix are initialized with magnitude 1.
    Second, a dense Layer with rule consequences is defined.
    Only weights defined in the classification_matrix are allowed to change,
    starting with magnitude 1.
    The weights are constrained to always be positive and sum up to 1 in the second dimension.
    That means that the weights connected to the same output node sum up to 1.
    Hence, the weights can be interpreted as the belief in this rule to lead to a certain output.
    """

    def __init__(
        self,
        n_features_input: int,
        n_membership_fcts: int,
        n_rules: int,
        n_output_classes: int,
        rule_matrix: Any = None,
        learn_conditions: bool = False,
        prune_weights: bool = False,
        softmax: torch.nn.Module = torch.nn.LogSoftmax(dim=1),
        classification_matrix: Any = None,
        learn_consequences: bool = True,
    ):
        """
        A 1D convolutional layer is used as a 2D dense layer to handle the multi-dimensionality of
        the previous layers. This allows an easier assignment of rules to the edge weights.

        Parameters
        ----------
        n_features_input : int
            Number of inputs before the Fuzzification layer.
        n_membership_fcts : int
            Number of membership functions for each input.
        n_rules : int
            Number of rules.
        n_output_classes: int
            Number of output classes.
        rule_matrix : np.Array
            Mask matrix determining how the weights are initialized.
            rule_matrix has to be of shape (n_rules, in_features, num_member_functions)
            If learn_conditions=True this matrix determines which weights are trained
        learn_conditions : bool
            Determines if the weights are allowed to be trained or should stay as initialized
        prune_weights : bool
            Determines if the weights are pruned to only allow the weights where rule_matrix == 1
            are allowed to be trained.
        softmax : torch.nn.Module
            softmax function that is used.
            Default: LogSoftmax (better performance with NLLLoss and Cross-Entropy Loss than Softmax)
        classification_matrix : np.Array
            Mask matrix determining how the weights are initialized. This models the consequences.
            classification_matrix has to be of shape (n_input_features, n_output_features)
        learn_consequences : bool
            Determines if the weights are allowed to be trained or should stay as initialized
        """

        super(FuzzyInference, self).__init__()
        self.eps = 1e-12

        ### Model the conditions
        self.conditions = torch.nn.Conv1d(
            n_features_input, n_rules, n_membership_fcts, bias=True
        )

        # Set if training of condition weights is allowed
        self.conditions.weight.requires_grad = learn_conditions

        # Infuse conditions into weights
        if rule_matrix is not None:
            self.rule_matrix = torch.from_numpy(
                np.where(rule_matrix == 0, 0, rule_matrix)
            )
            self.conditions.weight.data = deepcopy(self.rule_matrix)

        self.flat = torch.nn.Flatten()

        # Only weights that are initialized with 1 are allowed to be trained
        if prune_weights and learn_conditions:
            # pruning the weights of conditions according to the rule_matrix
            prune.custom_from_mask(self.conditions, "weight", self.rule_matrix)

        ### Model the consequences
        self.consequences = torch.nn.Linear(n_rules, n_output_classes, bias=True)
        # Set if training of consequences is allowed
        self.consequences.weight.requires_grad = learn_consequences

        self.softmax = softmax

        # Infuse consequences into weights
        if classification_matrix is not None:
            self.classification_matrix = torch.from_numpy(
                np.where(classification_matrix == 0, 0, classification_matrix)
            ).T
            self.consequences.weight.data = deepcopy(self.classification_matrix)
            # constraints
            prune.custom_from_mask(
                self.consequences, "weight", self.classification_matrix
            )
            self.consequences.register_backward_hook(
                self.clamp_and_normalize_consequences_backward
            )
            self.consequences.register_forward_pre_hook(
                self.clamp_and_normalize_consequences
            )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the FuzzyInference.
        First, a log transformation to the input is applied and followed by a
        Conv1D Layer. The result is flattened and an exp function is applied.
        Afterwards, the input is passed through a dense layer that models the consequences and
        finally a Softmax function is used in the end.

        Parameters
        ----------
        inputs : torch.Tensor
            input vector. Shape = (batchsize, n_features_input, n_membership_fcts)

        Returns
        -------
        output : torch.Tensor
            Interpretable prediction in classes. Shape = (batchsize, n_rules)
        """
        # Calculate the fullfillment of the conditions
        inputs = torch.log(inputs.type(torch.DoubleTensor))
        condition_fullfillment = self.conditions(inputs)
        condition_fullfillment = self.flat(condition_fullfillment)
        condition_fullfillment = torch.exp(condition_fullfillment)

        # Calculate the interpretable prediction with the consequences
        x_interp = self.consequences(condition_fullfillment)
        x_interp = self.softmax(x_interp)
        return x_interp

    def clamp_and_normalize_consequences_backward(self, grad_input, x, v) -> None:
        self.clamp_and_normalize_consequences(grad_input, x)

    def clamp_and_normalize_consequences(self, grad_input, x) -> None:
        self.clamp_consequences_weights()
        self.normalize_consequences_weights()

    def normalize_consequences_weights(self):
        normalizing_matrix = self.consequences.weight.sum(1, keepdim=True).clamp(
            min=self.eps
        )
        self.consequences.weight.data = self.consequences.weight / normalizing_matrix

    def clamp_consequences_weights(self):
        self.consequences.weight.data = self.consequences.weight.data.clamp(0, 2)
