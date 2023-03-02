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

import pandas as pd
import matplotlib.pyplot as plt
import os
import re
from torch.utils.data import Dataset
import torch


class DemonstratorDataSet(Dataset):
    def __init__(
        self,
        directory,
        sequence_length,
        window_size=None,
        header=None,
        drop=None,
        plot_data=False,
    ):
        super(DemonstratorDataSet, self).__init__()
        self.class_numbers = {}
        self.sequence_length = sequence_length
        self.window_size = window_size
        self.plot_data = plot_data
        if header is None:
            self.header = [
                "di_p_dif_load_peak_usr",
                "di_p_dif_load_usr",
                "di_p_dif_usr",
                "di_Power_act",
                "di_Power_mean",
                "di_v_act",
                "di_v_dif_usr",
                "di_v_ref",
                "diAI_0_Valve",
                "diAQ_0_Valve",
                "i_DEV_T_current",
                "i_M_Load",
                "i_M_overload",
                "i_PS_T_current",
                "r_I_act",
                "r_tq_act",
                "ui_DCOMStatus",
                "ui_LastError",
                "ui_LastWarning",
                "timestamp",
            ]
            self.drop = [
                "di_p_dif_load_peak_usr",
                "di_v_ref",
                "di_v_act",
                "ui_DCOMStatus",
                "ui_LastError",
                "ui_LastWarning",
                "timestamp",
                "diAI_0_Valve",
                "diAQ_0_Valve",
            ]
        else:
            self.header = header
            self.drop = drop
        self.data = []
        self.load_directory(directory)

    def load_csv(self, path):
        dataframe = pd.read_csv(filepath_or_buffer=path, sep=",", names=self.header)
        if self.drop is not None:
            dataframe = dataframe.drop(columns=self.drop)
        line_drops = len(dataframe.index) % self.sequence_length
        dataframe.drop(dataframe.tail(line_drops).index, inplace=True)
        m = re.search(r"E(\d)", path)
        err = m.group(1)
        label = int(err)

        if self.plot_data:
            for col in dataframe:
                plt.plot(range(len(dataframe.index)), dataframe[col], label=col)
            plt.legend()
            plt.show()
        return dataframe, label

    def load_directory(self, directory):
        for filename in os.listdir(directory):
            df, label = self.load_csv(os.path.join(directory, filename))
            if self.window_size is None:
                for i in range(len(df.index) // self.sequence_length):
                    self.data.append(
                        [
                            torch.from_numpy(
                                df.iloc[
                                    i * self.sequence_length : i * self.sequence_length
                                    + self.sequence_length
                                ].to_numpy()
                            ),
                            label,
                        ]
                    )
            else:
                for i in range(
                    (len(df.index) - self.sequence_length) // self.window_size
                ):
                    self.data.append(
                        [
                            torch.from_numpy(
                                df.iloc[
                                    i * self.window_size : i * self.window_size
                                    + self.sequence_length
                                ].to_numpy()
                            ),
                            label,
                        ]
                    )
            if label in self.class_numbers:
                self.class_numbers[label] += i
            else:
                self.class_numbers[label] = i

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return self.data[idx]

    def __len__(self):
        return len(self.data)
