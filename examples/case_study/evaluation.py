from torch import nn
import torch
import pandas as pd


class Evaluator:
    def __init__(self, model: nn.Module, forecast_horizon: int):
        self.losses = []
        self.model = model
        self.forecast_horizon = forecast_horizon
        self.loss_metric = nn.functional.mse_loss

    def evaluate(
        self, dataset: torch.utils.data.Dataset, index: int = None
    ) -> pd.DataFrame:
        self.model.eval()

        for features_past, target_past, target_future in dataset:
            features_past = features_past.unsqueeze(1)
            target_past = target_past.unsqueeze(1)

            with torch.no_grad():
                input = self.model.get_input(features_past, target_past)
                output = self.model(*input)
                forecasts = self.model.extract_forecasts(output)
            forecasts = forecasts.squeeze(1)
            if forecasts.size(-1) > 1:
                forecasts = forecasts[..., [index]]

            assert forecasts.shape == target_future.shape

            self.losses.append(
                self.loss_metric(
                    forecasts, target_future, reduction="none"
                ).flatten().tolist()
            )
        return pd.DataFrame(self.losses)
