from typing import Callable, Dict
import pandas as pd
import numpy as np
from models.imodel import IModel
from datasets.loader import Dataset, load_dataset
import auto_esn.utils.dataset_loader as dl
import torch.nn
from auto_esn.esn.esn import DeepESN
from auto_esn.esn.reservoir.initialization import CompositeInitializer, WeightInitializer
from auto_esn.esn.reservoir.activation import Activation, Identity


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

        self.n_steps_in = n_steps_in
        self.n_steps_out = n_steps_out
        self.metric_fun = metric
        
        self.dataset: Dataset = load_dataset(
            raw_dataset=dataset,
            n_steps_in=n_steps_in,
            n_steps_out=n_steps_out,
            test_fraction=test_frac
        )

        self.forecasts = None
        self.forecast_autoregressive = None

    def fit(self):
        i = CompositeInitializer()\
            .with_seed(23)\
            .sparse(density=0.1)\
            .spectral_noisy(spectral_radius=1.1, noise_magnitude=0.01)
            # .uniform()\
            # .regular_graph(4)\
            # .scale(0.9)\

        w = WeightInitializer()
        w.weight_hh_init = i
        esn = DeepESN(initializer = w, input_size=self.n_steps_in, hidden_size=400)
        esn.fit(self.dataset.train_x, self.dataset.train_y)

        self.forecasts = None
        self.forecast_autoregressive = None

    def __predict_single(self, shift: int) -> pd.Series:
        if self.forecasts:
            return self.forecasts[shift]

        # Predict on a test set batch -> multiple predictions shifted by one timestep
        predictions = np.squeeze(self.model.predict(self.dataset.test_x))

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
        init_window = np.expand_dims(self.dataset.train_x[-1, :], axis=0)

        # Acumulate the predictions in a NDArray
        predictions = np.array([[]])
        for _ in range(self.dataset.n_test() // self.n_steps_out):
            prediction = np.reshape(self.model.predict(init_window), (1, -1))
            init_window = np.concatenate((init_window[:, self.n_steps_out:],  prediction), axis=1)
            predictions = np.append(predictions, prediction)
        
        # Save forecasts and return the indexed Series
        self.forecast_autoregressive = pd.Series(
            predictions, index=range(
                self.dataset.n_train() - self.n_steps_out,
                self.dataset.n_train() + predictions.size - self.n_steps_out
            ))
        return self.forecast_autoregressive
    

    def predict(self):

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


