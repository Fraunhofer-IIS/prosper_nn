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

from .autoencoder import Autoencoder
from ..ecnn.ecnn_model import ECNN
import torch
from typing import Tuple


class EcnnAutoencoder(torch.nn.Module):
    """
    The module trains an on features encoded with an autoencoder.
    The autoencoder for the encoding can be trained parallel to the ECNN.
    """
    def __init__(self, autoencoder: Autoencoder, ecnn: ECNN) -> None:
        """
        Parameters
        ----------
        autoencoder : Autoencoder
            An autoencoder with a encoder and a decoder.
        ecnn : ECNN
            An ECNN that takes U and Y as input for the forward.

        Returns
        -------
        None
        """
        super(EcnnAutoencoder, self).__init__()
        self.autoencoder = autoencoder
        self.ecnn = ecnn

    def forward(
        self, U: torch.Tensor, Y: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        U : torch.Tensor
            The features that are encoded by the autoencoder that are afterwards
            passed to the ECNN.
        Y : torch.Tensor
            The target time series that is used for teacher forcing in the ECNN.

        Returns
        -------
        None
        """
        autoencoder_output = self.autoencoder(Y)
        autoencoder_compressed_Y = self.autoencoder.encode(Y)
        ecnn_output = self.ecnn(U, autoencoder_compressed_Y)
        ecnn_autoencoder_output = self.autoencoder.decode(ecnn_output)
        return ecnn_autoencoder_output, autoencoder_output, ecnn_output
