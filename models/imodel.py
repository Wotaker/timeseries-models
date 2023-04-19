import pandas as pd
from typing import Callable
from abc import ABC


class IModel(ABC):
    def __init__(self, dataset: pd.DataFrame, n_steps_in : int, n_steps_out : int, test_frac : float, metric : Callable, **params : dict ) -> None:
        pass

    def fit(self) -> None:
        pass

    def predict(self, n : int, autoreggressive : bool, shift : int = 0, **params : dict) -> pd.Series:
        pass

    def save(self, path : str) -> None:
        pass

    def load(self, path : str) -> None:
        pass