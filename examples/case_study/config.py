import pandas as pd

area = "output_and_income"  # "consumption_and_orders", "prices", "output_and_income"
n_evaluate_targets = 19

past_horizon = 24
forecast_horizon = 3
train_test_split_period = pd.Period("2011-01-01", freq="M")

# Model Training
batch_size = 32
n_epochs = 50
patience = 10

# Model Parameters
n_features_Y = 1
n_models = 25
