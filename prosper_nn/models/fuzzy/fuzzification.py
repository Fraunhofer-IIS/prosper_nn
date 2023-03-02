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
from .membership_block import MembershipBlock
from copy import deepcopy


class Fuzzification(torch.nn.Module):
    """
    Dynamically creates a layer of parallel MembershipBlocks.
    The number of blocks is determined by n_features_input, because it creates a MembershipBlock for
    each input.
    """

    def __init__(
        self, n_features_input: int, membership_fcts: dict, input_names: dict = None
    ):
        """
        Parameters
        ----------
        n_features_input : int
            Number of inputs.
        membership_fcts : Dict[str : torch.nn.Module]
            Set of member functions. Each MembershipBlock will contain these functions.
        input_names : List[str]
            Blocks will be named after this list for better debugging.
        """

        super(Fuzzification, self).__init__()
        if input_names is None:
            self.input_names = [
                "Block{}".format(count) for count in range(n_features_input)
            ]
        else:
            if not len(input_names) == n_features_input:
                raise ValueError(
                    "dimension mismatch: input_names must contain a name for any input"
                )
            self.input_names = input_names
        self.membership_fcts = membership_fcts
        for name in self.input_names:
            setattr(
                self,
                name,
                MembershipBlock(
                    membership_fcts=deepcopy(membership_fcts), block_name=name
                ),
            )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through all MemberBlocks.
        Parameters
        ----------
        inputs : torch.Tensor
            input vector of shape (batchsize, number inputs)

        Returns
        -------
        output : torch.Tensor
            Tensor of stacked block outputs. Shape (batchsize, n_features_input, n_membership_fcts)
        """
        inputs = torch.transpose(
            inputs, 1, 0
        )  # transpose input for better iteration capability
        output = {}
        for i, name in enumerate(self.input_names):
            output[name] = getattr(self, name)(inputs[i])
        output = torch.stack(tuple(output.values()), axis=1)
        return output
