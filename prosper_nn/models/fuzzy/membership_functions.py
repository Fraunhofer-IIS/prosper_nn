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
from typing import Any, Tuple, Callable


class GaussianMembershipFct(torch.autograd.Function):
    """
    Gaussian autograd function.
     * Fixed mean
     * Variable sigma parameter

    Class inherits from torch.autograd.Function.
    It has to reimplement the static methods forward and backward.
    See https://pytorch.org/tutorials/beginner/examples_autograd/two_layer_net_custom_function.html
    for more information and examples.
    """

    @staticmethod
    def forward(ctx: Any, inputs: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        """
        forward activation of the gaussian function

        Parameters
        ----------
        ctx : torch.Tensor
            context vector to save variables for the backward pass
        inputs : torch.Tensor
            input vector of shape (batchsize, 1)
        sigma : torch.Tensor (torch.nn.Parameter)
            learnable deviation parameter sigma

        Returns
        ----------
        output : torch.Tensor
            output of the gaussian activation function
        """
        output = torch.exp((-1 / 2) * torch.pow((inputs / sigma), 2))
        ctx.save_for_backward(inputs, sigma, output)

        return output

    @staticmethod
    def backward(
        ctx: Any, grad_outputs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        backward pass of the gaussian function

        Parameters
        ----------
        ctx : torch.Tensor
            context vector with saved variables of the forward pass
        grad_outputs : torch.Tensor
            Loss vector passed backward from previous layer

        Returns
        ----------
        (grad_input, grad_sigma) : Tuple(torch.Tensor, torch.Tensor)
            gradients w.r.t. inputs and w.r.t. sigma
        """
        inputs, sigma, output = ctx.saved_tensors
        grad_input = (output * (-1 * inputs / sigma) * (1 / sigma)) * grad_outputs
        grad_sigma = (
            output * (-1 * inputs / sigma) * (-1 * inputs / torch.pow(sigma, 2))
        ) * grad_outputs

        return grad_input, grad_sigma


class GaussianMembership(torch.nn.Module):
    """
    Gaussian member function

    * fixed mean at 0
    * deviation of the curve as learnable parameter sigma
    """

    def __init__(self, sigma_initializer: Callable = torch.nn.init.constant_) -> None:
        """

        Parameters
        ----------
        sigma_initializer : torch.nn.Initializer
        """
        super(GaussianMembership, self).__init__()
        self.sigma = torch.nn.Parameter(torch.Tensor(1))
        sigma_initializer(self.sigma, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """

        Parameters
        ----------
        inputs : torch.Tensor
            input vector of size (batchsize, 1)

        Returns
        -------
        torch.Tensor
        """
        return GaussianMembershipFct.apply(inputs, self.sigma)


class NormlogMembershipFct(torch.autograd.Function):
    """
    Normlog autograd function.
     * Fixed middle point at zero
     * Variable slope parameter

    Class inherits from torch.autograd.Function.
    It has to reimplement the static methods forward and backward.
    See https://pytorch.org/tutorials/beginner/examples_autograd/two_layer_net_custom_function.html
    for more information and examples.
    """

    @staticmethod
    def forward(
        ctx: Any, inputs: torch.Tensor, slope: torch.Tensor, negative: bool
    ) -> torch.Tensor:
        """forward activation of the normlog function

        Parameters
        ----------
        ctx : torch.Tensor
            context vector to save variables for the backward pass
        inputs : torch.Tensor
            input vector of shape (batchsize, 1)
        slope : torch.Tensor (torch.nn.Parameter)
            learnable slope parameter slope
        negative : bool
            determines if slope is negative or positive

        Returns
        ----------
        output : torch.Tensor
            output of the gaussian activation function

        slope parameter is restricted to the interval [0, inf] for negative=False
        and [-inf, 0] for negative=True
        """
        # negative
        if negative:
            c = torch.tensor(-1)
        # positive
        else:
            c = torch.tensor(1)
        # normlog activation function
        # ReLU activation as parameter restriction
        output = 1 / (1 + torch.exp(-4 * (c * torch.relu(slope) * inputs - 0.5)))
        ctx.save_for_backward(inputs, slope, output, c)

        return output

    @staticmethod
    def backward(
        ctx: Any, grad_outputs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """backward pass of the normlog function

        Parameters
        ----------
        ctx : torch.Tensor
            context vector with saved variables of the forward pass
        grad_outputs : torch.Tensor
            Loss vector passed backward from previous layer

        Returns
        ----------
        (grad_input, grad_sigma, None) : Tuple(torch.Tensor, torch.Tensor, None)
            gradients w.r.t. inputs and w.r.t. sigma
        """
        inputs, slope, output, c = ctx.saved_tensors
        # relu derivative
        d_relu = 1
        # relu derivative if relu<0
        if slope < 0:
            d_relu = 0
        # derivative of forward function w.r.t. input
        grad_input = (
            -1
            * torch.pow(output, 2)
            * torch.exp(-4 * c * torch.relu(slope) * inputs + 2)
            * (-4 * c * torch.relu(slope))
        ) * grad_outputs
        # and w.r.t slope
        grad_slope = (
            -1
            * torch.pow(output, 2)
            * torch.exp(-4 * c * torch.relu(slope) * inputs + 2)
            * (-4 * c * d_relu * inputs)
        ) * grad_outputs
        return grad_input, grad_slope, None


class NormlogMembership(torch.nn.Module):
    """
    Norm logistic function
    * fixed around zero
    * slope of the curve as trainable parameter slope
    """

    def __init__(
        self, negative: bool = False, slope_initializer: Callable = None
    ) -> None:
        """
        Parameters
        ----------
        negative : bool
            negates the slope of the function to be able to reuse the layer for falling values
        slope_initializer : torch.nn.Initializer
        """
        super(NormlogMembership, self).__init__()
        self.negative = negative
        self.slope = torch.nn.Parameter(torch.Tensor(1))
        if slope_initializer is None:
            torch.nn.init.constant_(self.slope, 1)
        else:
            slope_initializer(self.slope)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """

        Parameters
        ----------
        inputs : torch.Tensor
            input vector of size (batchsize, 1)

        Returns
        -------
            torch.Tensor
        """
        return NormlogMembershipFct.apply(inputs, self.slope, self.negative)
