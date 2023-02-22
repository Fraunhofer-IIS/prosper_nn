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

import numpy as np
import json
from os.path import exists
from typing import Tuple


class RuleManager:
    """
    JSON Reader to create the rule and classification matrix as an array.
    The matrix can be accessed via the object attributes.
    """

    def __init__(
        self,
        path: str,
        rule_matrix_shape: Tuple,
        classification_matrix_shape: Tuple = None,
    ):
        """

        Parameters
        ----------
        path : String
            path to the JSON file.
        rule_matrix_shape : Tuple
            Rule matrix of shape (n_rules, n_features_inputs, n_members)
        classification_matrix_shape : Tuple
            Classification matrix of shape (n_rules, n_features_output)
        """
        self.rule_matrix_shape = rule_matrix_shape
        self.classification_matrix_shape = classification_matrix_shape
        if exists(path):
            with open(path) as f:
                self.file = json.load(f)
            self.rules = self.file["rules"]
            self.input_names = self.file["input_names"]
            self.member_activations = self.file["member_activations"]
            self.classification_rules = self.file["classification_rules"]
            self.rule_matrix = self._create_rule_matrix()
            if self.classification_matrix_shape is not None:
                self.classification_matrix = self._create_classification_matrix()
        else:
            raise ValueError("path not found")

    def _create_rule_matrix(self):
        """
        Creates rule matrix from dictionaries

        Returns
        -------
        matrix : np.array
            Rule weight matrix of shape (n_rules, n_features_inputs, n_members)
        """
        matrix = np.zeros(self.rule_matrix_shape)
        for i, dic in enumerate(self.rules.values()):
            for name, activation in dic.items():
                matrix[i, self.input_names[name]] = self.member_activations[activation]
        return matrix

    def _create_classification_matrix(self):
        """
        Creates classification matrix from dictionaries

        Returns
        -------
        matrix : np.array
            Classification weight matrix of shape (n_rules, n_features_output)
        """
        matrix = np.zeros(self.classification_matrix_shape)
        for i, out_list in enumerate(self.classification_rules.values()):
            for index in out_list:
                matrix[i, index] = 1
        return matrix
