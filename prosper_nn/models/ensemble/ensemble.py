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

import warnings
import torch
from copy import deepcopy
from typing import Callable, Tuple, Union
import operator
import torch.nn.utils.prune as prune


def init_model(model: torch.nn.Module, init_func: Callable, *params, **kwargs) -> None:
    """
    Method to initialize all parameter of a given model.

    Parameters
    ----------
    model : torch.nn.Module
        PyTorch Module or Sequential that is duplicated to an ensemble
    init_func : Callable
        Function with which the weights of the model are initialized.
    *params : list, optional
        List with params for init_func.
    **kwargs : dict, optional
        Dict with kwargs for init_func.

    Returns
    -------
    None
    """

    for p in model.parameters():
        if len(p.shape) == 1:
            try:
                init_func(p, *params, **kwargs)
            except ValueError:
                warnings.warn(
                    "Bias could not be initialized with wished init function."
                    "Instead torch.nn.init.normal_ is chosen."
                )
                torch.nn.init.zeros_(p, *params, **kwargs)
        else:
            init_func(p, *params, **kwargs)


class Ensemble(torch.nn.Module):
    """
    Ensemble module to train multiple instances of a model.
    The different instances are initialized and saved as a class attribute model.

    The forward pass of this module returns one tensor that contains
    the outputs of the individual models at the beginning and their mean/median in the last index.

    To train the models iterate over the model output tensor and calculate and sum up the loss for each output.
    Call backward() for the summed up loss.

    The mean/median output can be used when the model is trained and in normal use.
    The mean/median output should not be used during training.

    """

    def __init__(
        self,
        model: torch.nn.Module,
        n_models: int,
        sparsity: int = 0.0,
        keep_pruning_mask: bool = False,
        initializer: Callable = torch.nn.init.kaiming_uniform_,
        parallel: bool = False,
        combination_type: str = "mean",
    ) -> None:
        """
        Parameters
        ----------
        model : torch.nn.Module
            PyTorch Module or Sequential that is duplicated to an ensemble
        n_models : int
            number of model instances that are created and trained.
        sparsity : float
            If model has pruned weights and the pruning mask should not be kept (see keep_pruning_mask),
            this value should be the amount the weight is pruned.
            The weights that were pruned in the original model are also pruned in the individual copies of the model in the ensemble.
            Each model in the ensemble is pruned independently.
        keep_pruning_mask : bool
            If True the pruning masks of the original model will be used for all the models in the ensemble.
        initializer : torch.nn.init
            Initializer used to initialize the different model instances.
            If None, all model instances are initialized the same.
        parallel : bool
            If True the forward path of the models are calculated in parallel.
            It is possible that this is not faster than without parallelization.
        combination_type : str
            In the ensemble the outputs of the individual models are combined.
            The final output can be the median or mean of the submodels outputs.

        Returns
        -------
        None
        """
        super(Ensemble, self).__init__()
        self.initializer = initializer
        orig_model = model
        self.n_models = n_models
        self.sparsity = sparsity
        self.keep_pruning_mask = keep_pruning_mask
        self.parallel = parallel
        self.combination_type = combination_type

        self._check_variables()

        self.models = torch.nn.ModuleList()
        if list(orig_model.named_buffers()):
            pruning = True
            assert (
                self.sparsity > 0.0
            ) != self.keep_pruning_mask, "It is either possible that sparsity > 0 or keep_pruning = True, but not both."
        else:
            pruning = False

        # Remove pruning on sparse matrices
        if pruning:
            pruned_weights = [
                (name, mask) for (name, mask) in orig_model.named_buffers()
            ]
            for name, mask in pruned_weights:
                pruned_object = operator.attrgetter(name[:-12])(orig_model)
                prune.remove(pruned_object, "weight")

        # setting class attributes for model instances
        for i in range(n_models):
            # Copy and initialize the models
            self.models.append(deepcopy(orig_model))
            if initializer is not None:
                # call initializer function to initialize each model
                init_model(self.models[i], self.initializer)

            # Apply pruning to the models if the original model had pruned weights
            if pruning:
                for name, mask in pruned_weights:
                    object_to_prune = operator.attrgetter(name[:10])(self.models[i])
                    if self.keep_pruning_mask:
                        prune.custom_from_mask(object_to_prune, "weight", mask)
                    elif self.sparsity > 0:
                        prune.random_unstructured(
                            object_to_prune, "weight", self.sparsity
                        )
                    else:
                        ValueError(
                            "If the orig_model has pruned weights, sparsity should be greater than 0 "
                            "or keep_pruning_mask should be True."
                        )

    def forward(self, *input: Union[torch.Tensor, Tuple[torch.Tensor]]) -> torch.Tensor:

        """
        Forward passing function of the module. Passes input through all instances of the
        given model and returns a torch.Tensor with an additional dimension.
        It contains the individual outputs and as last entry their mean/median.

        Parameters
        ----------
        input : Union[torch.Tensor, Tuple[torch.Tensor]]
            Input vector as torch.Tensor or a Tuple of torch.Tensor

        Returns
        -------
        torch.Tensor
            Tensor containing the individual model outputs in the first n_models entries of the
            first dimension and their mean/median in the last entry.
        """

        if self.parallel:
            futures = [torch.jit.fork(model, *input) for model in self.models]
            outs = [torch.jit.wait(fut) for fut in futures]
        else:
            outs = [model(*input) for model in self.models]

        outs = torch.stack(outs)
        if self.combination_type == "mean":
            combined_output = torch.mean(outs, dim=0, keepdim=True)
        elif self.combination_type == "median":
            combined_output = torch.median(outs, dim=0, keepdim=True)[0]
        return torch.cat((outs, combined_output), dim=0)

    def _check_variables(self) -> None:
        """
        Checks if self.combination_type has valid input.
        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        if self.combination_type not in ["mean", "median"]:
            raise ValueError(
                '"{}" is not a valid type for combination. '
                'It must be either "mean" or "median".'.format(self.combination_type)
            )

    def set(self, variable: str, value) -> None:
        """
        Set for all models in the ensemble a variable to the given value.
        Parameters
        ----------
        variable : str
            The name of the variable in the submodels that should be set.
        value
            The value the variable is set to.

        Returns
        -------
        None
        """
        for i in range(self.n_models):
            setattr(self.models[i], variable, value)
