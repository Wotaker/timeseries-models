from typing import Tuple, List
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pandas as pd


@dataclass
class Dataset:
    dates: pd.Series
    series_all: pd.Series
    series_train: pd.Series
    series_test: pd.Series
    train_x: npt.NDArray
    train_y: npt.NDArray
    test_x: npt.NDArray
    test_y: npt.NDArray

    def n_values(self) -> int:
        return self.series_all.size
    
    def n_train(self) -> int:
        return self.series_train.size
    
    def n_test(self) -> int:
        return self.series_test.size


def read_dataset(df: pd.DataFrame, target_column: str | None = None) -> Tuple[pd.Series, pd.Series]:
    """
    Tries to extract from the input Dataframe the total values (target column values of the analyzed timeseries)
    and the date labels.
    
    IMPORTANT! This function assumes that the input Dataframe is loaded without index column!

    Parameters
    ----------
    df : pd.DataFrame
        The input, raw pandas Dataframe
    target_column : str | None
        The name of the target column, if None, takes the last non-date column, by default None

    Returns
    -------
    Tuple[total: pd.Series, dates: pd.Series]
        The total values and the corresponding dates
    """

    # Preprocess data
    if "DATE" in list(map(lambda x: x.upper(), df.keys())):

        # Preprocess only if date column is present
        date_idx = list(map(lambda x: x.upper(), df.keys())).index("DATE")
        df.rename(columns={df.keys()[date_idx]: 'Date'}, inplace=True)
        df['Date'] = pd.to_datetime(df['Date'])                 # Always format the date
        df = df.sort_values(['Date'])                           # Sort by date
    else:

        df['Date'] = 0

    # Extract timeseries from original dataframe
    total_column = target_column if target_column else df.keys().difference(['Date'])[-1]
    print(f"Target column: '{total_column}'")
    total = df[total_column]
    dates = df['Date']

    return total, dates


def train_test_split(dataset: pd.Series, test_fraction=0.1) -> Tuple[pd.Series, pd.Series]:

    # Assign the count variables
    all_obs = dataset.shape[0]
    test_obs = int(np.round(test_fraction * all_obs))
    train_obs = all_obs - test_obs

    # Split the dataset
    ds_train = dataset[:train_obs]
    ds_test = dataset[train_obs:]

    # Test split
    assert (dataset == pd.concat((ds_train, ds_test))).values.all(), "Train-Test split incorrect!"

    return ds_train, ds_test


def split_sequence_supervised(
        sequence: npt.NDArray,
        n_steps_in: int,
        n_steps_out: int,
        test_split: bool = False,
        shuffle: bool = False) -> Tuple[npt.NDArray, npt.NDArray]:
    
    # Get the start/cut index
    sequence_len = sequence.shape[0]
    n_steps_cycle = n_steps_in + n_steps_out
    n_cut = sequence_len % n_steps_cycle
    
    # Cut the sequence
    sequence = sequence[:-n_cut] if test_split else sequence[n_cut:]
    sequence_len = sequence.shape[0]
    
    # Create buckets
    x, y = list(), list()
    for i in range(sequence_len - n_steps_cycle + 1):

        start_x_idx = i
        start_y_idx = start_x_idx + n_steps_in

        x.append(sequence[start_x_idx:start_y_idx])
        y.append(sequence[start_y_idx:start_y_idx+n_steps_out])
    
    # Cast to NDArray
    x, y = np.array(x), np.array(y)

    # Shuffle
    if shuffle:
        np.random.shuffle(x)
        np.random.shuffle(y)

    return x, y


def load_dataset(
    raw_dataset: pd.DataFrame,
    n_steps_in: int,
    n_steps_out: int = 1,
    test_fraction: float = 0.1,
    target_column: str | None = None
) -> Dataset:
    
    ds, dates = read_dataset(raw_dataset, target_column)
    ds_train, ds_test = train_test_split(ds, test_fraction)

    train_x, train_y = split_sequence_supervised(ds_train, n_steps_in, n_steps_out, test_split=False, shuffle=False)
    test_x,  test_y  = split_sequence_supervised(ds_test,  n_steps_in, n_steps_out, test_split=True,  shuffle=False)

    assert (ds_test[:n_steps_in].values == test_x[0]).all(),                       "Train-test-split is incorrect!"
    assert (ds_test[n_steps_in:n_steps_in+n_steps_out].values == test_y[0]).all(), "Train-test-split is incorrect!"

    dataset = Dataset(
        dates,
        ds,
        ds_train,
        ds_test,
        train_x,
        train_y,
        test_x,
        test_y
    )

    return dataset

