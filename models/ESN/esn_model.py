from typing import Callable, Dict, Tuple, List
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np
from models.imodel import IModel
from datasets.loader import Dataset, load_dataset
import auto_esn.utils.dataset_loader as dl
import torch.nn
from auto_esn.esn.esn import DeepESN
from auto_esn.esn.reservoir.initialization import CompositeInitializer, WeightInitializer
from auto_esn.esn.reservoir.activation import Activation, Identity

import optuna



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


class ESNModel(IModel):
    
    def __init__(
            self, 
            dataset: pd.DataFrame,
            n_steps_in: int,
            n_steps_out: int,
            test_frac: float,
            metric: Callable,
            **params: Dict
        ) -> None:
        self._device = torch.device('cpu')

        self.n_steps_in = n_steps_in
        self.n_steps_out = n_steps_out
        self.metric_fun = metric
        
        self.dataset: Dataset = load_dataset(
            raw_dataset=dataset,
            n_steps_in=n_steps_in,
            n_steps_out=n_steps_out,
            test_fraction=test_frac
        )
        xlabel = params["xlabel"] if "xlabel" in params.keys() else None
        ylabel = params["ylabel"] if "ylabel" in params.keys() else None
        self.plotter = Plotter(self.dataset.series_all, xlabel, ylabel)


        self.forecasts = None
        self.forecast_autoregressive = None

    def tune(self, n_trials=50):
        def objective(trial):
            density = trial.suggest_float('density', 0, 1)
            spectral_radius = trial.suggest_float('spectral_radius' ,0.5, 1.5)
            noise_magnitude = trial.suggest_float('noise_magnitude', 0, 0.1)
            hidden_size = trial.suggest_int('hidden_size', 50, 1000)
            i = CompositeInitializer()\
            .sparse(density=density)\
            .spectral_noisy(spectral_radius=spectral_radius, noise_magnitude=noise_magnitude)
            # .uniform()\
            # .regular_graph(4)\
            # .scale(0.9)\
             # .with_seed(23)\

            w = WeightInitializer()
            w.weight_hh_init = i
            self.model = DeepESN(initializer = w, input_size=self.n_steps_in, hidden_size=hidden_size)
            train_x = torch.from_numpy(self.dataset.train_x).to(self._device)
            train_y = torch.from_numpy(self.dataset.train_y).to(self._device)
            self.model.fit(train_x, train_y)
            self.forecasts = None
            self.forecast_autoregressive = None
            forecast = self.predict(50, autoreggressive=True)
            rmse = np.sqrt(np.mean((forecast - self.dataset.series_all[forecast.index][:50])**2))
            return rmse
        
        study = optuna.create_study()
        study.optimize(objective, n_trials=n_trials)
        return study.best_params.values()

    def fit(self, density=0.1, spectral_radius=1.1, noise_magnitude=0.01, hidden_size=400):
        i = CompositeInitializer()\
            .sparse(density=density)\
            .spectral_noisy(spectral_radius=spectral_radius, noise_magnitude=noise_magnitude)
            # .uniform()\
            # .regular_graph(4)\
            # .scale(0.9)\
            # .with_seed(23)\
        w = WeightInitializer()
        w.weight_hh_init = i
        self.model = DeepESN(initializer = w, input_size=self.n_steps_in, hidden_size=hidden_size)
        train_x = torch.from_numpy(self.dataset.train_x).to(self._device)
        train_y = torch.from_numpy(self.dataset.train_y).to(self._device)
        self.model.fit(train_x, train_y)

        self.forecasts = None
        self.forecast_autoregressive = None

    def __predict_single(self, shift: int) -> pd.Series:
        if self.forecasts:
            return self.forecasts[shift]

        # Predict on a test set batch -> multiple predictions shifted by one timestep
        predictions = np.squeeze(self.model(torch.from_numpy(self.dataset.test_x)).to(self._device))

        # Transform the 2D predictions into a list of Series
        series_list = list()
        for prediction_idx in range(predictions.shape[0]):

            series_start_idx = self.dataset.n_train() + self.n_steps_in + prediction_idx
            series_end_idx = self.dataset.n_train() + self.n_steps_in + prediction_idx + self.n_steps_out
            series_list.append(pd.Series(
                data=predictions[prediction_idx],
                index=range(series_start_idx, series_end_idx)
            ))
        
        # Save forecasts and return the desired shifted Series
        self.forecasts = series_list
        return self.forecasts[shift]
    

    def __predict_autoregressive(self) -> pd.Series:

        if self.forecast_autoregressive is not None:
            return self.forecast_autoregressive
        
        # Creates the window upon which a prediction takes place
        init_window = torch.from_numpy(np.expand_dims(self.dataset.train_x[-1, :], axis=0)).to(self._device)

        # Acumulate the predictions in a NDArray
        predictions = np.array([[]])
        for _ in range(self.dataset.n_test() // self.n_steps_out):
            prediction = np.reshape(self.model(init_window), (1, -1))
            init_window = torch.from_numpy(np.concatenate((init_window[:, self.n_steps_out:],  prediction), axis=1)).to(self._device)
            predictions = np.append(predictions, prediction)
        
        # Save forecasts and return the indexed Series
        self.forecast_autoregressive = pd.Series(
            predictions, index=range(
                self.dataset.n_train() - self.n_steps_out,
                self.dataset.n_train() + predictions.size - self.n_steps_out
            ))
        return self.forecast_autoregressive
    

    def predict(self, n: int, autoreggressive: bool, shift: int = 0, **params: dict) -> pd.Series:

        # Get the forecast
        forecast = self.__predict_autoregressive() if autoreggressive else self.__predict_single(shift)

        if n == -1:
            # Return a whole forecast
            return forecast
        
        # Return only the n first from the forecast
        n = min(n, forecast.size)
        start_idx = forecast.index[0]
        end_idx = start_idx + n - 1

        return forecast.loc[start_idx:end_idx]



if __name__ == '__main__':
    MyESN_100_30 = ESNModel(
    dataset=pd.read_csv("datasets/MackeyGlass.csv"),
    n_steps_in=300,
    n_steps_out=1,
    test_frac=0.1,
    metric=None,
    )
    # params = MyESN_100_30.tune()
    # MyESN_100_30.fit(*params)
    MyESN_100_30.fit()
    forecast_autoregressive = MyESN_100_30.predict(-1, autoreggressive=True)
    MyESN_100_30.plotter.plot_forecast(forecast_autoregressive)
