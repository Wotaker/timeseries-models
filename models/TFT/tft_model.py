from typing import Callable, Dict, Tuple, List

from models.imodel import IModel
import copy
from pathlib import Path
import warnings
from datasets.loader import load_dataset, Dataset
import numpy.typing as npt
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

import numpy as np
import pandas as pd
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger
import torch

from pytorch_forecasting import Baseline, TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import SMAPE, PoissonLoss, QuantileLoss
from pytorch_forecasting.models.temporal_fusion_transformer.tuning import optimize_hyperparameters
import pickle

import warnings

warnings.filterwarnings("ignore")

class Plotter():

    def __init__(self, dataset: pd.Series, xlabel, ylabel) -> None:
        self.dataset_series = dataset
        self.metrices_map = {
            "RMSE": self.__get_RMSE,
            "MAPE": self.__get_MAPE
        }
        self.xlabel = xlabel
        self.ylabel = ylabel

    def __get_RMSE(self, y: pd.Series, y_hat: pd.Series, window: int = 10) -> Tuple[str, float, pd.Series]:
        
        rmse = np.sqrt(np.mean((y_hat - y)**2))
        rolling_rmse = y.rolling(window).apply(
            lambda x: np.sqrt(mean_squared_error(x, y_hat[x.index])),
            raw=False
        )

        return "RMSE", rmse, rolling_rmse


    def __get_MAPE(self, y: pd.Series, y_hat: pd.Series, window: int = 10) -> Tuple[str, float, pd.Series]:
        
        err = y_hat - y
        mape_vals = (np.abs((err) / y) * 100)
        mape = np.mean(mape_vals[mape_vals < np.inf])
        rolling_mape = y_hat.rolling(window).apply(
            lambda y: np.mean(np.abs(((err) / y) * 100)),
            raw=False
        )

        return "MAPE", mape, rolling_mape


    def plot_forecasts(
        self,
        forecast_series: List[pd.Series],
        xlim: Tuple[int, int] | None = None,
        ylim: Tuple[int, int] | None = None,
        metric_name: str = "RMSE",
        linestyle: str = 'None'
    ):

        metric_func = self.metrices_map[metric_name.upper()]

        # Scatter original data
        plt.plot(self.dataset_series.index, self.dataset_series.values, color='tab:blue', label='Original',
                 marker='.', markersize=2, linestyle=linestyle)

        metric_vals = list()
        for series in forecast_series:

            # Calculate metrices
            metric_name, metric, metric_series = metric_func(y=self.dataset_series[series.index], y_hat=series)
            metric_vals.append(metric)
            
            # Scatter forecast
            plt.plot(series.index, series.values, color="tab:orange", alpha=0.1,
                     marker='.', markersize=2, linestyle=linestyle)

            # Plot evaluation metric
            plt.plot(metric_series.index, metric_series.values, color="red", alpha=0.1, lw=1)
        
        metric_aggregated = np.mean(np.array(metric_vals))

        # Legend hack
        plt.plot([-10], [0], color="tab:orange", label='Prediction', marker='.', markersize=2, linestyle=linestyle)
        plt.plot([-10, -9], [0, 0], color="red", label=metric_name, alpha=0.5, lw=1)

        plt.legend()
        plt.xlabel(self.xlabel)
        plt.ylabel(self.ylabel)
        plt.title(f"Sumaric forecast results\n{metric_name}: {round(metric_aggregated, 3)}")
        plt.xlim(xlim if xlim else (self.dataset_series.index[0], self.dataset_series.index[-1]))
        plt.ylim(ylim) if ylim else plt.ylim(0, self.dataset_series.max() * 1.2)
        plt.show()
    
    
    def plot_forecast(
        self,
        forecast_series: pd.Series,
        xlim: Tuple[int, int] | None = None,
        ylim: Tuple[int, int] | None = None,
        metric_name: str = "RMSE",
        linestyle: str = 'None'
    ):

        # Calculate metrices
        metric_func = self.metrices_map[metric_name.upper()]
        metric_name, metric, metric_series = metric_func(y=self.dataset_series[forecast_series.index], y_hat=forecast_series)

        # Scatter original data
        plt.plot(self.dataset_series.index, self.dataset_series.values, color='tab:blue', label='Original',
                 marker='.', markersize=2, linestyle=linestyle)

        # Scatter forecast
        plt.plot(forecast_series.index, forecast_series.values, color="tab:orange", label='Prediction',
                 marker='.', markersize=2, linestyle=linestyle)

        # Plot evaluation metric
        plt.plot(metric_series.index, metric_series.values, color="red", label=metric_name, alpha=0.5)

        plt.legend()
        plt.xlabel(self.xlabel)
        plt.ylabel(self.ylabel)
        plt.title(f"{metric_name}: {round(metric, 3)}")
        plt.xlim(xlim)
        plt.ylim(ylim) if ylim else plt.ylim(0, self.dataset_series.max() * 1.2)
        plt.show()
    
    def plot_dataset(
        self,
        xlim: Tuple[int, int] | None = None,
        ylim: Tuple[int, int] | None = None,
        linestyle: str = 'None',
        save_path: str | None = None
    ):
        
        # Scatter original data
        plt.plot(self.dataset_series.index, self.dataset_series.values, color='tab:blue',
                 marker='.', markersize=2, linestyle=linestyle)
        
        plt.xlabel(self.xlabel)
        plt.ylabel(self.ylabel)
        plt.xlim(xlim)
        plt.ylim(ylim) if ylim else plt.ylim(0, self.dataset_series.max() * 1.2)
        plt.savefig(save_path, bbox_inches='tight') if save_path else None
        plt.show()
        

class TFTModel(IModel):
    def __init__(
        self,
        dataset: pd.DataFrame,
        n_steps_in: int,
        n_steps_out: int,
        test_frac: float,
        metric: Callable,
        **params: Dict
    ) -> None:
        
        self.n_steps_in= n_steps_in
        self.n_steps_out = n_steps_out
        self.metric_fun = metric
        self.dataset: Dataset = load_dataset(
            raw_dataset= dataset,
            n_steps_in= n_steps_in,
            n_steps_out= n_steps_out,
            test_fraction= test_frac,
            target_column=params["target_column"] if "target_column" in params.keys() else None
        )
        
        xlabel = params["xlabel"] if "xlabel" in params.keys() else None
        ylabel = params["ylabel"] if "ylabel" in params.keys() else None

        self.plotter = Plotter(self.dataset.series_all, xlabel, ylabel)
        
        df_test = pd.DataFrame()
        df_test["target"] = self.dataset.series_test
        df_test["time_idx"] = self.dataset.series_test.index
        df_test["const"] = 1
        self.df_test = df_test
        
        df_train = pd.DataFrame()
        df_train["target"] = self.dataset.series_train
        df_train["time_idx"] = self.dataset.series_train.index
        df_train["const"] = 1
        self.df_train = df_train
    
        training_cutoff = self.df_train["time_idx"].max() - n_steps_out        
        self.train_time_series_dataset = TimeSeriesDataSet(
            self.df_train[lambda x: x.time_idx <= training_cutoff],
            time_idx = "time_idx",
            target = "target",
            group_ids = ["const"],
            min_encoder_length = n_steps_in,
            max_encoder_length = n_steps_in,
            min_prediction_length=1,
            max_prediction_length=n_steps_out,
            time_varying_known_reals=["time_idx"],
            time_varying_unknown_reals=["target"],
            add_relative_time_idx=True,
            add_target_scales=True,
            add_encoder_length=True
        )
        validation = TimeSeriesDataSet.from_dataset(self.train_time_series_dataset, self.df_train, predict=True, stop_randomization=True)
        batch_size = 64
        self.train_dataloader = self.train_time_series_dataset.to_dataloader(train=True, batch_size=batch_size)
        self.val_dataloader = validation.to_dataloader(train=False, batch_size=batch_size)
        
    def fit(self, **params) -> None:
        training_params = params.get("training_params", {})
        tft = TemporalFusionTransformer.from_dataset(
            self.train_time_series_dataset,
            learning_rate=training_params.get("learning_rate", 0.05),
            hidden_size=training_params.get("hidden_size", 8),
            attention_head_size=training_params.get("attention_head_size", 2),
            dropout=training_params.get("dropout", 0.1),
            hidden_continuous_size=training_params.get("hidden_continuous_size", 2),
            loss=self.metric_fun,
            log_interval=10,
            reduce_on_plateau_patience=4
        )
        self.model = tft

        early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=params["min_delta"] if "min_delta" in params.keys() else 1e-4, patience=10, verbose=False, mode="min")
        lr_logger = LearningRateMonitor()  # log the learning rate
        logger = TensorBoardLogger("lightning_logs")
        
        trainer = pl.Trainer(
            max_epochs=params.get("max_epochs", 50),
            enable_model_summary=False,
            gradient_clip_val=training_params.get("gradient_clip_val", 0.68),
            limit_train_batches=params.get("limit_train_batches", 50),
            callbacks=[lr_logger, early_stop_callback],
            logger=logger
        )
        
        trainer.fit(
            self.model,
            train_dataloaders=self.train_dataloader,
            val_dataloaders=self.val_dataloader
        )
        
        best_model_path = trainer.checkpoint_callback.best_model_path
        self.model = TemporalFusionTransformer.load_from_checkpoint(best_model_path)
    def optuna_fit(self, max_epochs, n_trials, model_path):
        # create study
        study = optimize_hyperparameters(
            self.train_dataloader,
            self.val_dataloader,
            model_path=model_path,
            n_trials=n_trials,
            max_epochs=max_epochs,
            gradient_clip_val_range=(0.01, 1.0),
            hidden_size_range=(1, 100),
            hidden_continuous_size_range=(1, 100),
            attention_head_size_range=(1, 6),
            learning_rate_range=(0.001, 0.2),
            dropout_range=(0.05, 0.5),
            reduce_on_plateau_patience=4,
            use_learning_rate_finder=False,
        )
            
        return study.best_trial.params
        
    def __moving_window(self, window):
        dataset = TimeSeriesDataSet(
            window,
            time_idx = "time_idx",
            target = 'target',
            group_ids = ["const"],
            min_encoder_length = self.n_steps_in,
            max_encoder_length = self.n_steps_in,
            min_prediction_length=1,
            max_prediction_length=1,
            time_varying_known_reals=["time_idx"],
            time_varying_unknown_reals=["target"],
            add_relative_time_idx=True,
            add_target_scales=True,
            add_encoder_length=True
        )
        prediction = self.model.predict(dataset, mode="prediction", return_x=False)
        return prediction[0][0]

    def __predict_autoreggressive(self, window: pd.DataFrame, n: int):
        res = []
        time_idx_start = window["time_idx"].max()
        for i in range(n):
            new_element = pd.DataFrame.from_dict({int(time_idx_start + i + 1): [0, time_idx_start + i + 1, 1]}, orient = "index", columns = ["target", "time_idx", "const"])
            window = pd.concat([window, new_element])
            result = self.__moving_window(window)
            result = result.item()
            res.append(result)
            window.at[time_idx_start + i + 1, "target"] = result
            window = window.tail(-1)
        return pd.Series(data=res,
                         index=range(time_idx_start + 1, time_idx_start + n + 1))
    
    def __predict_not_autoreggressive(self, dataframe: pd.DataFrame, n: int):
        time_idx_start = dataframe["time_idx"].max()
        for i in range(n):
            new_element = pd.DataFrame.from_dict({int(time_idx_start + i + 1): [0, time_idx_start + i + 1, 1]}, orient = "index", columns = ["target", "time_idx", "const"])
            dataframe = pd.concat([dataframe, new_element])
        
        data = TimeSeriesDataSet(
            dataframe,
            time_idx = "time_idx",
            target = 'target',
            group_ids = ["const"],
            min_encoder_length = self.n_steps_in,
            max_encoder_length = self.n_steps_in,
            min_prediction_length=n,
            max_prediction_length=n,
            time_varying_known_reals=["time_idx"],
            time_varying_unknown_reals=["target"],
            add_relative_time_idx=True,
            add_target_scales=True,
            add_encoder_length=True
        )
        result = self.model.predict(data, mode="prediction", return_x=False)
        return pd.Series(data=result[0],
                         index=range(time_idx_start + 1, time_idx_start + n + 1))
        
    def predict(self, n : int, autoreggressive : bool, shift : int = 0, **params : dict) -> pd.Series:
        start = self.df_test["time_idx"].min() + shift
        max_idx = self.df_test["time_idx"].max()
        if start + self.n_steps_in > max_idx:
            raise Exception("Test dataset is too small for step that large!")
        predict_base = (self.df_test[lambda x: x.time_idx >= start])[lambda x: x.time_idx < start + self.n_steps_in]
        if autoreggressive:
            if n == -1:
                n = max_idx - start - self.n_steps_in
            result = self.__predict_autoreggressive(predict_base, n)
        else:
            result = self.__predict_not_autoreggressive(predict_base, n)
        return result

    def save(self, path : str) -> None:
        torch.save(self.model.state_dict(), path)

    def load(self, path : str) -> None:
        self.model = TemporalFusionTransformer.load_from_checkpoint(path)