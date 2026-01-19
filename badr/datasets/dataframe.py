from typing import List, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from badr.datasets import Dataset


def _preprocess_dataframe(
    df: pd.DataFrame, target_col: str, sensitive_cols: Union[str, List[str]]
) -> Tuple[pd.DataFrame, np.ndarray, List[np.ndarray]]:
    # Load a DataFrame and split it into features and target.
    # Check if sensitive_cols is a string or a list
    if isinstance(sensitive_cols, str):
        sensitive_cols = [sensitive_cols]

    # Split the DataFrame into features and target
    X = df.drop(columns=[target_col] + sensitive_cols)
    y = df[target_col].to_numpy()
    for sensitive_col in sensitive_cols:
        if sensitive_col not in df.columns:
            raise ValueError(
                f"Sensitive column {sensitive_col} not found in DataFrame."
            )
    sensitive_features = [
        np.asarray(df[sensitive_col].values) for sensitive_col in sensitive_cols
    ]

    return X, y, sensitive_features


def _partition_sensitive(
    X: np.ndarray,
    sensitive_features: List[np.ndarray],
    n_groups: int = 2,
    safeguard: bool = True,
) -> List[np.ndarray]:
    # Partition data by arbitrary sensitive feature combinations.
    # stack sensitive feature arrays into an (n_samples, k) matrix
    sf_arrs = [np.asarray(arr).ravel() for arr in sensitive_features]
    sf_matrix = np.stack(sf_arrs, axis=1)
    # find unique combinations and assign each sample a group index
    unique_vals, inverse = np.unique(sf_matrix, axis=0, return_inverse=True)

    n_features = X.shape[1]
    buckets: List[np.ndarray] = []
    # collect groups that have more samples than features
    for grp_idx in range(len(unique_vals)):
        idx = np.where(inverse == grp_idx)[0]
        threshold = n_features if safeguard else 1
        if idx.size >= threshold:
            buckets.append(idx)

    if len(buckets) < n_groups:
        raise ValueError(
            f"Cannot form {n_groups} groups; only {len(buckets)} buckets available"
        )

    # merge smallest buckets until we have exactly n_groups
    while len(buckets) > n_groups:
        sizes = [b.size for b in buckets]
        i1, i2 = np.argsort(sizes)[:2]
        b1 = buckets.pop(max(i1, i2))
        b2 = buckets.pop(min(i1, i2))
        buckets.append(np.concatenate([b1, b2]))

    # slice out each bucket and return as (X_numpy, y_numpy) tuples
    return buckets


def load_dataframe(
    df: pd.DataFrame,
    target_col: str,
    sensitive_cols: Union[str, List[str]],
    name: str = "user dataframe",
    n_groups: int = 2,
    test_size: float = 0.3,
    random_state: int = 42,
) -> Dataset:
    """
    Build a :class:`~badr.datasets.Dataset` from a user-provided DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        Input data.
    target_col : str
        Target column name.
    sensitive_cols : str or list[str]
        Sensitive column(s) used to form groups. These columns are not included in X.
    name : str, default="user dataframe"
        Dataset name stored on the returned Dataset.
    n_groups : int, default=2
        Number of groups to return (by merging sensitive-value buckets if needed).
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
        If any `sensitive_cols` are not present in `df`, or if `n_groups` cannot be formed.
    """
    # Placeholder for actual implementation
    X, y, sensitive_features = _preprocess_dataframe(df, target_col, sensitive_cols)
    # STRATIFICATION BEGIN
    n_samples, n_features = X.shape[0], X.shape[1]
    sf_arrs = [np.asarray(arr).ravel() for arr in sensitive_features]
    sf_matrix = np.stack(sf_arrs, axis=1)
    rows = [tuple(row) for row in sf_matrix]
    unique_vals, inverse = np.unique(rows, return_inverse=True)

    n_features = X.shape[1]
    buckets: List[np.ndarray] = []
    # collect groups that have more samples than features
    for grp_idx in range(len(unique_vals)):
        idx = np.where(inverse == grp_idx)[0]
        if idx.size > n_features:
            buckets.append(idx)

    while len(buckets) > n_groups:
        sizes = [b.size for b in buckets]
        i1, i2 = np.argsort(sizes)[:2]
        b1 = buckets.pop(max(i1, i2))
        b2 = buckets.pop(min(i1, i2))
        buckets.append(np.concatenate([b1, b2]))
    strata = np.full(n_samples, -1, dtype=int)
    for gid, idx in enumerate(buckets):
        strata[idx] = gid

    mask = strata >= 0
    Xf, yf, strataf = X.iloc[mask], y[mask], strata[mask]
    # STRATIFICATION END

    # split X, y, and each sensitive array into train/test
    all_splits = train_test_split(
        Xf,
        yf,
        *sensitive_features,
        test_size=test_size,
        stratify=strataf,
        random_state=random_state,
    )
    # unpack first four: X_train, X_test, y_train, y_test
    X_train, X_test, y_train, y_test, *rest = all_splits
    # rest = [sf1_train, sf1_test, sf2_train, sf2_test, ...]
    sens_train = rest[0::2]  # take every even: sf*_train
    sens_test = rest[1::2]  # take every odd: sf*_test

    # scale features
    scaler = StandardScaler()
    scaler.set_output(transform="pandas")
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # partition by sensitive features
    group_idx = _partition_sensitive(X_train, sens_train, n_groups)

    group_idx_test = _partition_sensitive(X_test, sens_test, n_groups)
    return Dataset(name, X_train, X_test, y_train, y_test, group_idx, group_idx_test)
