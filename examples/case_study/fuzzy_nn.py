from pathlib import Path
import torch

from prosper_nn.models.fuzzy.rule_manager import RuleManager

from prosper_nn.models.fuzzy.membership_functions import (
    NormlogMembership,
    GaussianMembership,
)
from prosper_nn.models.fuzzy.fuzzification import Fuzzification
from prosper_nn.models.fuzzy.fuzzy_inference import FuzzyInference
from prosper_nn.models.fuzzy.defuzzification import Defuzzification


class Benchmark_Fuzzy_NN(torch.nn.Module):
    """
    Construct Fuzzy Neural Network and add methods to run it in unified benchmark pipeline.
    """

    multivariate = False

    def __init__(self, n_features_input, past_horizon, n_rules=3, n_output_classes=9):
        super(Benchmark_Fuzzy_NN, self).__init__()
        self.past_horizon = past_horizon
        membership_fcts = {
            "decrease": NormlogMembership(negative=True),
            "constant": GaussianMembership(sigma=1.0),
            "increase": NormlogMembership(),
        }
        n_membership_fcts = len(membership_fcts)

        fuzzification = Fuzzification(
            n_features_input=n_features_input, membership_fcts=membership_fcts
        )

        rule_manager = RuleManager(
            path=Path(__file__).parent / "fuzzy_rules.json",
            rule_matrix_shape=(n_rules, n_features_input, n_membership_fcts),
            classification_matrix_shape=(n_rules, n_output_classes),
        )
        fuzzy_inference = FuzzyInference(
            n_features_input=n_features_input,
            n_rules=n_rules,
            n_output_classes=n_output_classes,
            n_membership_fcts=n_membership_fcts,
            rule_matrix=rule_manager.rule_matrix,
            prune_weights=False,
            learn_conditions=True,
            classification_matrix=rule_manager.classification_matrix,
        )

        defuzzification = Defuzzification(n_output_classes, n_features_output=3)

        self.fuzzy = torch.nn.Sequential(
            fuzzification, fuzzy_inference, defuzzification
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        output = self.fuzzy(X)
        return output

    def get_input(self, features_past, target_past):
        return (features_past[-1],)

    def extract_forecasts(self, ensemble_output):
        mean = ensemble_output[-1]
        return mean.T.unsqueeze(2)
