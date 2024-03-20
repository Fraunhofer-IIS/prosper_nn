# %%
import sys
import os

sys.path.append(os.path.abspath("../.."))
sys.path.append(os.path.abspath(".."))
sys.path.append(os.path.abspath("."))


# %%

import torch
import pandas as pd
import numpy as np
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

from prosper_nn.utils import visualize_forecasts, sensitivity_analysis
from fredmd import Dataset_Fredmd

from config import past_horizon, forecast_horizon, train_test_split_period


# %% Select target
target = "DNDGRG3M086SBEA"

# %%
fredmd_test = Dataset_Fredmd(
    past_horizon,
    forecast_horizon,
    split_date=train_test_split_period,
    data_type="test",
    target=target,
)

# %% Visualization of the ECNN ensemble Heatmap

ecnn_ensemble = torch.load(Path(__file__).parent / f"ECNN_{target}.pt")

rolling_origin_start_date = pd.Period("2011-01-01", freq="M")
ecnn_ensemble.eval()
features_past, target_past, target_future = fredmd_test.get_one_rolling_origin(
    rolling_origin_start_date
)
target_past = target_past.unsqueeze(1)
features_past = features_past.unsqueeze(1)

with torch.no_grad():
    forecasts = ecnn_ensemble(features_past, target_past)

forecasts = forecasts[:-1, past_horizon:, 0]
forecasts = fredmd_test.rescale(forecasts, rolling_origin_start_date)
forecasts = fredmd_test.postprocess(forecasts, rolling_origin_start_date)

torch.save(forecasts, Path(__file__).parent / "forecasts.pt")

# %% Figure 6: Uncertainty Heatmap

matplotlib.rcParams.update({"font.size": 15})

start_point = torch.tensor(
    fredmd_test.original_data.loc[rolling_origin_start_date - 1, fredmd_test.target]
)
visualize_forecasts.heatmap_forecasts(
    forecasts.squeeze(2),
    start_point=start_point,
    sigma=0.25,
    num_interp=30,
    window_height=200,
    xlabel="Forecast step",
    save_at=Path(__file__).parent / f"heatmap_{target}.pdf",
    title="Heatmap ensemble forecast",
)


# %% Figure 7: Sensitivity Analysis of the Input Features
matplotlib.rcParams.update({"font.size": 12})

all_ros_features_past, all_ros_targets_past, _ = fredmd_test.get_all_rolling_origins()
all_ros_features_past = all_ros_features_past.transpose(1, 0).unsqueeze(2)
all_ros_targets_past = all_ros_targets_past.transpose(1, 0).unsqueeze(2)

for model in ecnn_ensemble.models:
    model.batchsize = 1

sensitivity = sensitivity_analysis.calculate_sensitivity_analysis(
    ecnn_ensemble,
    *(all_ros_features_past, all_ros_targets_past),
    output_neuron=(-1, past_horizon + 1, 0, 0),
    batchsize=1,
)

restricted_sensitivity_matrix = sensitivity[:, -1].squeeze(1)
labels = fredmd_test.features + [fredmd_test.target]

fig = sns.heatmap(
    restricted_sensitivity_matrix.T,
    center=0,
    cmap="coolwarm",
    robust=True,
    cbar_kws={"label": r"$\frac{\partial\ output}{\partial\ input}$"},
    vmin=-torch.max(abs(restricted_sensitivity_matrix)),
    vmax=torch.max(abs(restricted_sensitivity_matrix)),
    rasterized=True,
)
plt.xlabel("Rolling Origins")
plt.ylabel("Features")
plt.yticks(ticks=0.5 + np.arange(len(labels)), labels=labels, rotation=0)
plt.title(f"Sensitivity of '{fredmd_test.target}'s one step forecast")
plt.grid(visible=False)
plt.tight_layout()
plt.savefig(Path(__file__).parent / f"sensitivity_{target}.pdf")
