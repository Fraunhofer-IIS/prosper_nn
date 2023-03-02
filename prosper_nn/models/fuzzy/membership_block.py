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


class MembershipBlock(torch.nn.Module):
    """
    Block of a given set of membership functions
    Input/Output = 1 / number of membership functions
    """

    def __init__(self, membership_fcts: dict, block_name: str = None):
        """
        Parameters
        ----------
        membership_fcts : Dict[str : torch.nn.Module]
            set of membership functions
        block_name : str
            name of the block
        """
        super(MembershipBlock, self).__init__()
        if block_name is not None:
            self.block_name = block_name
        # creating a class attribute for each membership function
        for name, function in membership_fcts.items():
            setattr(self, name, function)  # set attributes
        self.membership_fcts = list(membership_fcts.keys())  # save names

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the block.
        Giving the same input to all membership functions.

        Parameters
        ----------
        inputs : torch.Tensor
            input vector of size (batchsize, 1)

        Returns
        -------
        output : torch.Tensor
            stacked outputs of all membership functions. Shape (batchsize, len(membership_fcts))
        """
        output = {}
        for name in self.membership_fcts:
            output[name] = getattr(self, name)(inputs)  # output of membership function

        output = torch.stack(tuple(output.values()), axis=1)
        return output
