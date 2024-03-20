from pathlib import Path
import torch
import pandas as pd
import numpy as np
from typing import Tuple


class Dataset_Fredmd(torch.utils.data.Dataset):
    """
    Creates a PyTorch suitable data set for FRED-MD. The data is transformed to
    log-differences and rolling origins are created. Afterwards each rolling origin
    is scaled independently and data is split into training, validation and test.
    """

    def __init__(
        self,
        past_horizon: int,
        forecast_horizon: int,
        split_date: pd.Period,
        data_type: str = "test",
        target: str = "CPIAUCSL",
    ):
        assert data_type in ["train", "val", "test"]
        self.past_horizon = past_horizon
        self.forecast_horizon = forecast_horizon
        self.window_size = past_horizon + forecast_horizon
        self.split_date = split_date

        # Select variables from "prices" group without 'OILPRICEx'
        self.features = [
            "WPSFD49207",
            "WPSFD49502",
            "WPSID61",
            "WPSID62",
            # "OILPRICEx",
            "PPICMM",
            "CPIAUCSL",
            "CPIAPPSL",
            "CPITRNSL",
            "CPIMEDSL",
            "CUSR0000SAC",
            "CUSR0000SAD",
            "CUSR0000SAS",
            "CPIULFSL",
            "CUSR0000SA0L2",
            "CUSR0000SA0L5",
            "PCEPI",
            "DDURRG3M086SBEA",
            "DNDGRG3M086SBEA",
            "DSERRG3M086SBEA",
        ]
        self.target = target
        self.original_data = self.get_data()
        self.n_rolling_origins = len(self.original_data) - self.window_size

        df = self.preprocess(self.original_data)
        rolling_origins = self.get_rolling_origins(df)
        self.mean, self.std = self.get_scales(rolling_origins)
        df_train, df_val, df_test = self.train_test_split(
            rolling_origins,
        )
        if data_type == "train":
            self.df = df_train
        elif data_type == "val":
            self.df = df_val
        else:
            self.df = df_test

    def __len__(self) -> int:
        return self.df.index.get_level_values(0).nunique()

    def get_scales(
        self, rolling_origins: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        mean = rolling_origins.groupby("rolling_origin_start_date").apply(
            lambda x: x.head(self.past_horizon).mean()
        )
        std = rolling_origins.groupby("rolling_origin_start_date").apply(
            lambda x: x.head(self.past_horizon).std()
        )
        assert (
            (std != 0).all().all()
        ), "Standard deviation is zero and will lead to NaNs"
        return mean, std

    def split_past_future(
        self, timeseries: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        past = timeseries.head(self.past_horizon)
        future = timeseries.tail(self.forecast_horizon)
        return past, future

    def scale(
        self,
        past: pd.DataFrame,
        future: pd.DataFrame,
        rolling_origin_start_date: pd.Period,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        mean = self.mean.loc[rolling_origin_start_date]
        std = self.std.loc[rolling_origin_start_date]
        past = (past - mean) / std
        future = (future - mean) / std
        return past, future

    def rescale(
        self, forecast: torch.Tensor, rolling_origin_start_date: pd.Period
    ) -> torch.Tensor:
        mean = self.mean.loc[rolling_origin_start_date, self.target]
        std = self.std.loc[rolling_origin_start_date, self.target]
        mean = torch.tensor(mean)
        std = torch.tensor(std)
        forecast = forecast * std + mean
        return forecast

    def get_one_rolling_origin(
        self, rolling_origin_start_date: pd.Period
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        timeseries = self.df.loc[rolling_origin_start_date]
        past, future = self.split_past_future(timeseries)
        past, future = self.scale(past, future, rolling_origin_start_date)

        assert past.notnull().all().all()
        assert future.notnull().all().all()

        features_past = torch.tensor(past[self.features].values)
        target_past = torch.tensor(past[self.target].values).unsqueeze(1)
        target_future = torch.tensor(future[self.target].values).unsqueeze(1)
        return features_past, target_past, target_future

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        rolling_origin_start_date = self.df.index.get_level_values(0).unique()[idx]
        features_past, target_past, target_future = self.get_one_rolling_origin(
            rolling_origin_start_date
        )
        return features_past, target_past, target_future

    def get_data(self) -> pd.DataFrame:
        path = Path(__file__).parent
        df = pd.read_csv(
            path / "2024-01.csv",
            parse_dates=["sasdate"],
            index_col="sasdate",
            usecols=["sasdate", self.target] + self.features,
        )
        df = df.drop("Transform:")
        df.index = pd.PeriodIndex(df.index, freq="M")
        return df

    def get_rolling_origins(self, df: pd.DataFrame) -> pd.DataFrame:
        rolling_origins = [
            df.iloc[i : i + self.window_size] for i in range(self.n_rolling_origins)
        ]
        rolling_origins = pd.concat(
            rolling_origins, keys=df.index[: self.n_rolling_origins]
        )
        rolling_origins.index.rename("rolling_origin_start_date", level=0, inplace=True)
        return rolling_origins

    def train_test_split(
        self, rolling_origins: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        # Avoid data leakage
        last_date_val_rolling_origin = self.split_date - self.window_size
        last_date_train_rolling_origin = last_date_val_rolling_origin - 12
        df_train = rolling_origins.loc[:last_date_train_rolling_origin]
        df_val = rolling_origins.loc[
            last_date_train_rolling_origin
            + self.forecast_horizon : last_date_val_rolling_origin
        ]
        df_test = rolling_origins.loc[
            last_date_val_rolling_origin + self.forecast_horizon :
        ]
        return df_train, df_val, df_test

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.apply(np.log).diff()
        df = df.iloc[1:]
        return df

    def postprocess(
        self, forecast: torch.Tensor, rolling_origin_start_date: pd.Period
    ) -> torch.Tensor:
        start_value = self.original_data.loc[rolling_origin_start_date, self.target]
        start_value = torch.tensor(start_value)
        forecast = start_value * torch.exp(torch.cumsum(forecast, dim=1))
        return forecast

    def get_all_rolling_origins(
        self,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        all_targets_past = []
        all_targets_future = []
        all_features_past = []
        for idx in range(self.__len__()):
            features_past, target_past, target_future = self.__getitem__(idx)
            all_targets_past.append(target_past)
            all_targets_future.append(target_future)
            all_features_past.append(features_past)
        all_targets_past = torch.stack(all_targets_past, dim=1)
        all_targets_future = torch.stack(all_targets_future, dim=1)
        all_features_past = torch.stack(all_features_past, dim=1)
        return all_features_past, all_targets_past, all_targets_future
