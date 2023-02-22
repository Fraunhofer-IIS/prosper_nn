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


class Defuzzification(torch.nn.Module):
    """
    Defuzzification step in a Fuzzy Neural Network. It translates back the interpretable prediction
    into a numerical prediction. This is done by a affine transformation (torch.nn.Linear layer).
    """

    def __init__(
        self,
        n_output_classes: int,
        n_features_output: int = 1,
    ) -> None:
        """

        Parameters
        ----------
        n_output_classes : int
            The number of output classes in the previous FuzzyInference step.
        n_features_output : int
            Number of target features.
            Default: 1
        """

        super(Defuzzification, self).__init__()
        self.defuzzification = torch.nn.Linear(
            in_features=n_output_classes, out_features=n_features_output
        )

    def forward(self, x_interp: torch.Tensor) -> torch.Tensor:
        """
        Apply a Linear layer.

        Parameters
        ----------
        x_interp : torch.Tensor
            input vector of size (batchsize, n_output_classes)

        Returns
        -------
        output : torch.Tensor
            Output  after defuzzification as the numerical prediction.
            Shape (batchsize, n_features_output)
        """

        output = self.defuzzification(x_interp)
        return output
