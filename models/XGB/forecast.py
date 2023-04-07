from typing import Tuple, List
from dataclasses import dataclass
from collections import defaultdict

import numpy as np
import numpy.typing as npt
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb


def window_forecast(past_data_batch: npt.NDArray, pretrained_model: xgb.XGBRegressor) -> npt.NDArray:

    return np.squeeze(pretrained_model.predict(past_data_batch))


def ar_forecast(
    train_x,
    pretrained_model: xgb.XGBRegressor,
    test_ds_size: int
) -> npt.NDArray[np.float64]:
    
    init_window = np.expand_dims(train_x[-1], axis=0)
    predictions = np.array([])
    for _ in range(test_ds_size):
        
        prediction = np.expand_dims(pretrained_model.predict(init_window), axis=0)
        init_window = np.concatenate((init_window[:, 1:],  prediction), axis=1)
        predictions = np.append(predictions, prediction[0, 0])
    
    return predictions


def get_forecast_series(forecast: npt.NDArray, train_ds_size: int, n_steps_in: int, n_steps_out: int) -> List[pd.Series]:

    series_list = list()
    for forecast_idx in range(forecast.shape[0]):

        series_start_idx = train_ds_size + n_steps_in + forecast_idx
        series_end_idx = train_ds_size + n_steps_in + forecast_idx + n_steps_out
        series_list.append(pd.Series(
            data=forecast[forecast_idx],
            index=range(series_start_idx, series_end_idx)
        ))
    
    return series_list


def get_forecast_series_ar(forecast: npt.NDArray, train_ds_size: int, test_ds_size) -> pd.Series:
    return pd.Series(forecast, index=range(train_ds_size, train_ds_size + test_ds_size))


def plot_forecasts(dataset_series: pd.Series, forecast_series: List[pd.Series], xlim: Tuple[int, int]):

    plt.scatter(dataset_series.index, dataset_series.values, color='tab:blue', label='Original', s=2)

    for series in forecast_series:
        plt.scatter(series.index, series.values, color="tab:orange", s=2, alpha=0.1)

    plt.scatter([-10], [0], color="tab:orange", label='Prediction', s=2, alpha=0.1)
    plt.legend()
    plt.xlabel("Month index")
    plt.ylabel("Monthly Mean Total Sunspot Number")
    plt.title("Sumaric forecast results")
    plt.xlim(xlim)
    plt.show()


def plot_forecast(dataset_series: pd.Series, forecast_series: List[pd.Series], idx: int, xlim: Tuple[int, int]):
    
    suffix = defaultdict(lambda: "th", {1: "st", 2: "nd", 3: "rd"})


    plt.scatter(dataset_series.index, dataset_series.values, color='tab:blue', label='Original', s=2)
    plt.scatter(forecast_series[idx].index, forecast_series[idx].values, color="tab:orange", label='Prediction', s=2)

    plt.legend()
    plt.xlabel("Month index")
    plt.ylabel("Monthly Mean Total Sunspot Number")
    plt.title(f"Forecast since {idx + 1}{suffix[idx + 1]} data patch")
    plt.xlim(xlim)
    plt.show()


def plot_forecast_ar(dataset_series: pd.Series, forecast_series: pd.Series, xlim: Tuple[int, int], n_steps_in: int):
    
    plt.scatter(dataset_series.index, dataset_series.values, color='tab:blue', label='Original', s=2)
    plt.scatter(forecast_series.index, forecast_series.values, color="tab:orange", label='Prediction', s=2)

    plt.legend()
    plt.xlabel("Month index")
    plt.ylabel("Monthly Mean Total Sunspot Number")
    plt.title(f"Autoregressive forecast for {n_steps_in} values lookbehind")
    plt.xlim(xlim)
    plt.show()
