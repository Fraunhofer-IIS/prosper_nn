# %%
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

# %%
dir = Path(__file__).parent
areas = ["output_and_income", "consumption_and_orders", "prices"]
df_collection = []
for area in areas:
    df = pd.read_csv(dir / f"overall_losses_{area}.csv", index_col=[0, 1, 2])
    df.index.set_names(["var", "model", "rolling_origin"], inplace=True)
    df.columns = range(1, len(df.columns) + 1)
    df.columns.name = "Forecast Step"
    df_collection.append(df)

# %% Overall results
dfs = pd.concat(df_collection, keys=areas, axis=0)
dfs.groupby(level=(2)).mean().round(3).to_latex(dir / f"overall_results.tex")

# %% Results for first step over three variable groups
forecast_step = 1

dfs = pd.concat([df[forecast_step] for df in df_collection], keys=areas, axis=1)
mean_error_per_group = dfs.groupby(level=1).mean()

# Plot
ax = mean_error_per_group.rank().plot(kind='bar', figsize=(10, 4))
ax.spines[['right', 'top']].set_visible(False)
plt.ylabel('Rank')
plt.xlabel('Model')
plt.legend(bbox_to_anchor=(1,1))
plt.tight_layout()
plt.savefig(dir / 'model_ranking_in_groups.pdf')