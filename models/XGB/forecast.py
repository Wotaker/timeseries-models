from typing import Tuple, List
from dataclasses import dataclass
from collections import defaultdict

from sklearn.metrics import mean_squared_error

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


def get_RMSE(y: pd.Series, y_hat: pd.Series, window: int = 10) -> Tuple[str, float, pd.Series]:
    
    rmse = np.sqrt(np.mean((y_hat - y)**2))
    rolling_rmse = y.rolling(window).apply(
        lambda x: np.sqrt(mean_squared_error(x, y_hat[x.index])),
        raw=False
    )

    return "RMSE", rmse, rolling_rmse


def get_MAPE(y: pd.Series, y_hat: pd.Series, window: int = 10) -> Tuple[str, float, pd.Series]:
    
    err = y_hat - y
    mape_vals = (np.abs((err) / y) * 100)
    mape = np.mean(mape_vals[mape_vals < np.inf])
    rolling_mape = y_hat.rolling(10).apply(
        lambda y: np.mean(np.abs(((err) / y) * 100)),
        raw=False
    )

    return "MAPE", mape, rolling_mape


METRICES_MAP = {
    "RMSE": get_RMSE,
    "MAPE": get_MAPE
}


def plot_forecasts(
    dataset_series: pd.Series,
    forecast_series: List[pd.Series],
    xlim: Tuple[int, int],
    metric_name: str = "RMSE"
):

    metric_func = METRICES_MAP[metric_name.upper()]

    # Scatter original data
    plt.scatter(dataset_series.index, dataset_series.values, color='tab:blue', label='Original', s=2)

    metric_vals = list()
    for series in forecast_series:

        # Calculate metrices
        metric_name, metric, metric_series = metric_func(y=dataset_series[series.index], y_hat=series)
        metric_vals.append(metric)
        
        # Scatter forecast
        plt.scatter(series.index, series.values, color="tab:orange", s=2, alpha=0.1)

        # Plot evaluation metric
        plt.plot(metric_series.index, metric_series.values, color="red", alpha=0.1, lw=1)
    
    metric_aggregated = np.mean(np.array(metric_vals))

    # Legend hack
    plt.scatter([-10], [0], color="tab:orange", label='Prediction', s=2)
    plt.plot([-10, -9], [0, 0], color="red", label=metric_name, alpha=0.5, lw=1)

    plt.legend()
    plt.xlabel("Month index")
    plt.ylabel("Monthly Mean Total Sunspot Number")
    plt.title(f"Sumaric forecast results\n{metric_name}: {round(metric_aggregated, 3)}")
    plt.xlim(xlim)
    plt.ylim(-20, dataset_series.max() * 1.2)
    plt.show()


def plot_forecast(
    dataset_series: pd.Series,
    forecast_series: List[pd.Series],
    idx: int,
    xlim: Tuple[int, int],
    metric_name: str = "RMSE"
):
    
    suffix = defaultdict(lambda: "th", {1: "st", 2: "nd", 3: "rd"})

    # Calculate metrices
    metric_func = METRICES_MAP[metric_name.upper()]
    metric_name, metric, metric_series = metric_func(y=dataset_series[forecast_series[idx].index], y_hat=forecast_series[idx])

    # Scatter original data
    plt.scatter(dataset_series.index, dataset_series.values, color='tab:blue', label='Original', s=2)

    # Scatter forecast
    plt.scatter(forecast_series[idx].index, forecast_series[idx].values, color="tab:orange", label='Prediction', s=2)

    # Plot evaluation metric
    plt.plot(metric_series.index, metric_series.values, color="red", label=metric_name, alpha=0.5)

    plt.legend()
    plt.xlabel("Month index")
    plt.ylabel("Monthly Mean Total Sunspot Number")
    plt.title(f"Forecast since {idx + 1}{suffix[idx + 1]} data patch\n{metric_name}: {round(metric, 3)}")
    plt.xlim(xlim)
    plt.ylim(-20, dataset_series.max() * 1.2)
    plt.show()


def plot_forecast_ar(
    dataset_series: pd.Series,
    forecast_series: pd.Series,
    xlim: Tuple[int, int],
    n_steps_in: int,
    metric_name: str = "RMSE"
):

    # Calculate metrices
    metric_func = METRICES_MAP[metric_name.upper()]
    metric_name, metric, metric_series = metric_func(y=dataset_series[forecast_series.index], y_hat=forecast_series)
    
    # Scatter original data
    plt.scatter(dataset_series.index, dataset_series.values, color='tab:blue', label='Original', s=2)

    # Scatter forecast
    plt.scatter(forecast_series.index, forecast_series.values, color="tab:orange", label='Prediction', s=2)

    # Plot evaluation metric
    plt.plot(metric_series.index, metric_series.values, color="red", label=metric_name, alpha=0.5)

    plt.legend()
    plt.xlabel("Month index")
    plt.ylabel("Monthly Mean Total Sunspot Number")
    plt.title(f"Autoregressive forecast for {n_steps_in} values lookbehind\n{metric_name}: {round(metric, 3)}")
    plt.xlim(xlim)
    plt.ylim(-20, dataset_series.max() * 1.2)
    plt.show()
