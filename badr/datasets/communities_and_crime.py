from os import makedirs, path
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from badr.datasets import BADR_PATH, Dataset


def _download_communities_and_crime(replace: bool = False) -> pd.DataFrame:
    # Download the Communities and Crime dataset from OpenML.
    data_dir = path.join(BADR_PATH, "communitiesandcrime")
    makedirs(data_dir, exist_ok=True)
    ds = fetch_openml(
        name="us_crime",
        version=1,
        as_frame=True,
        data_home=data_dir,
        cache=not replace,
    )
    return ds.frame


def _preprocess_communities_and_crime(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, List[str]]:
    # Preprocess the Communities and Crime dataset.
    # drop ID and fold columns
    df = df.drop(["state", "county", "community", "communityname", "fold"], axis=1)
    # filter out sparse columns and rows
    df = df.replace("?", np.nan)
    df = df[df.columns[df.isna().mean() < 0.1]]
    df = df.replace("?", np.nan).dropna(axis=0, how="any")
    df = df.astype(float)

    # target
    y = df["ViolentCrimesPerPop"].astype(np.float32).to_numpy()
    # sensitive attribute: race = argmax of race percentages
    race_cols = ["racepctblack", "racePctWhite", "racePctAsian", "racePctHisp"]
    race = np.argmax(df[race_cols].values, axis=1)

    # drop target and race cols
    df = df.drop(columns=race_cols + ["ViolentCrimesPerPop"])

    # one-hot any remaining categoricals (none expected)
    X = pd.get_dummies(df, prefix_sep="_", dummy_na=False, dtype=np.float32)
    num_cols = list(X.columns)

    return X, y, race, num_cols


def _partition_sensitive(
    X: pd.DataFrame, race_vals: np.ndarray, n_groups: int
) -> List[np.ndarray]:
    # Partition X,y into n_groups by race categories, merging if needed.
    idx = np.arange(len(X))
    race_vals = np.array(race_vals)
    race_cats = np.unique(race_vals)

    groups = [idx[race_vals == r] for r in race_cats]
    groups.sort(key=lambda g: len(g), reverse=True)

    if n_groups > len(groups):
        raise ValueError(
            f"Cannot form {n_groups} groups; only {len(groups)} distinct races present."
        )

    # merge smallest until desired number
    while len(groups) > n_groups:
        sizes = [len(g) for g in groups]
        i1, i2 = np.argsort(sizes)[:2]
        g1 = groups.pop(max(i1, i2))
        g2 = groups.pop(min(i1, i2))
        groups.append(np.concatenate([g1, g2]))
    return groups


def fetch_communities_and_crime(
    n_groups: int = 2, test_size: float = 0.3, random_state: int = 42
) -> Dataset:
    """
    Fetch Communities and Crime (OpenML ``us_crime``, v1).

    Target: ``ViolentCrimesPerPop`` (float).
    Groups: race category from ``argmax`` over {black, white, asian, hispanic} race
    percentage columns, merged until `n_groups` remain. Features are standardized
    on the training split.

    Parameters
    ----------
    n_groups : int, default=2
        Number of groups to return.
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
        If `n_groups` is larger than the number of race categories present.
    """
    df = _download_communities_and_crime(replace=False)
    df = df.dropna()

    X, y, race, num_cols = _preprocess_communities_and_crime(df)

    # stratify train/test by race
    X_train, X_test, y_train, y_test, race_train, race_test = train_test_split(
        X,
        y,
        race,
        test_size=test_size,
        stratify=race,
        random_state=random_state,
    )

    scaler = StandardScaler()
    X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
    X_test[num_cols] = scaler.transform(X_test[num_cols])

    group_idx = _partition_sensitive(X_train, race_train, n_groups)
    X_train = X_train.to_numpy()
    group_idx_test = _partition_sensitive(X_test, race_test, n_groups)
    X_test = X_test.to_numpy()

    return Dataset(
        "Communities and Crime",
        X_train,
        X_test,
        y_train,
        y_test,
        group_idx,
        group_idx_test,
    )
