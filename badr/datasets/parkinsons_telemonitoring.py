from os import makedirs, path
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from badr.datasets import BADR_PATH, Dataset, fetch_csv

CAT, NUM = "CAT", "NUM"


def _download_parkinsons(replace: bool = False) -> pd.DataFrame:
    # Download the Parkinson's Telemonitoring dataset via sklearn.fetch_openml.
    data_dir = path.join(BADR_PATH, "parkinsons_telemonitoring")
    makedirs(data_dir, exist_ok=True)
    df = fetch_csv(
        url="https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/telemonitoring/parkinsons_updrs.data",
        data_home=data_dir,
        replace=replace,
        sep=",",
    )
    return df


def _preprocess_parkinsons(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    # Filter, map target, extract sex, split NUM/CAT columns.
    # initial column type spec
    df.dropna(axis=0)
    y = np.asarray(df["total_UPDRS"].values)
    df = df.drop(["motor_UPDRS", "total_UPDRS"], axis=1)
    X = df
    original_sex = np.asarray(X["sex"].values)

    return X, y, original_sex


def _partition_sensitive(
    X: np.ndarray, sex_vals: np.ndarray, n_groups: int
) -> List[np.ndarray]:
    # Partition X, y into two groups by sex flag only.
    idx = np.arange(len(X))
    groups = [
        idx[sex_vals == 0],
        idx[sex_vals == 1],
    ]
    if n_groups != 2:
        raise ValueError(
            f"Parkinson's Telemonitoring supports only n_groups=2, got {n_groups}"
        )
    return groups


def fetch_parkinsons(
    n_groups: int = 2, test_size: float = 0.3, random_state: int = 42
) -> Dataset:
    """
    Fetch Parkinson's Telemonitoring (UCI ``parkinsons_updrs.data``).

    Target: ``total_UPDRS``.
    Groups: ``sex`` (0 vs 1). Only `n_groups=2` is supported.
    Features are standardized on the training split.

    Parameters
    ----------
    n_groups : int, default=2
        Must be 2.
    test_size : float, default=0.3
        Test split size.
    random_state : int, default=42
        Train/test split seed.

    Returns
    -------
    Dataset
        Train/test splits and per-group indices.

    Raises
    ------
    ValueError
        If `n_groups` is not 2.
    """
    df = _download_parkinsons(replace=False)
    X, y, sex = _preprocess_parkinsons(df)

    # stratify by sex only
    X_train, X_test, y_train, y_test, sex_train, sex_test = train_test_split(
        X,
        y,
        sex,
        test_size=test_size,
        stratify=sex,
        random_state=random_state,
    )

    # scale numeric cols on train, apply to test
    scaler = StandardScaler()
    scaler.set_output(transform="pandas")
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    group_idx = _partition_sensitive(X_train, sex_train, n_groups)
    X_train = X_train.to_numpy()  # type: ignore
    group_idx_test = _partition_sensitive(X_test, sex_test, n_groups)
    X_test = X_test.to_numpy()
    return Dataset(
        "Parkinson's Telemonitoring",
        X_train,
        X_test,
        y_train,
        y_test,
        group_idx,
        group_idx_test,
    )
