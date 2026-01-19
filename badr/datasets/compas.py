from itertools import product
from os import makedirs, path
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from badr.datasets import BADR_PATH, Dataset, fetch_csv


def _download_compas(replace: bool = False) -> pd.DataFrame:
    # Download the Compas dataset from OpenML.
    data_dir = path.join(BADR_PATH, "compas")
    makedirs(data_dir, exist_ok=True)
    df = fetch_csv(
        url="https://raw.githubusercontent.com/propublica/compas-analysis/master/compas-scores-two-years.csv",
        data_home=data_dir,
        replace=replace,
        sep=",",
        index_col=0,
    )
    return df


def _preprocess_compas(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    # Preprocess the Compas dataset.
    # target
    df = df[df["is_recid"] != -1]
    df = df[df["days_b_screening_arrest"] <= 30]
    df = df[df["days_b_screening_arrest"] >= -30]
    df = df[df["c_charge_degree"] != "O"]
    y = df["two_year_recid"].to_numpy().astype(np.float32)
    race = df["race"].to_numpy()
    race_map = {
        "African-American": 0,
        "Caucasian": 1,
        "Asian": 2,
        "Other": 3,
        "Hispanic": 4,
        "Native American": 5,
    }
    race = np.array([race_map[r] for r in race]).astype(np.float32)
    X = df[
        [
            "age",
            "priors_count",
            "juv_fel_count",
            "juv_misd_count",
            "juv_other_count",
        ]
    ]
    X.insert(column="c_charge_degree", value=df["c_charge_degree"] == "M", loc=0)
    sex = df["sex"] == "Female"
    sex = sex.to_numpy().astype(np.float32)
    X.insert(
        column="days_in_jail",
        value=(
            pd.to_datetime(df["c_jail_out"]) - pd.to_datetime(df["c_jail_in"])
        ).dt.days,
        loc=0,
    )
    X.insert(
        column="days_in_custody",
        value=(
            pd.to_datetime(df["out_custody"]) - pd.to_datetime(df["in_custody"])
        ).dt.days,
        loc=0,
    )
    X = X.astype(np.float32)
    return X, y, sex, race


def _partition_sensitive(
    X: np.ndarray,
    sex_vals: np.ndarray,
    race_vals: np.ndarray,
    n_groups: int,
) -> List[np.ndarray]:
    # Partition X,y into n_groups by race categories, merging if needed.
    n_features = X.shape[1]
    sex_cats = np.unique(sex_vals)
    race_cats = np.unique(race_vals)
    all_combinations = list(product(sex_cats, race_cats))

    combinations = []
    for s, r in all_combinations:
        idx = np.where((sex_vals == s) & (race_vals == r))[0]
        threshold = n_features
        if idx.size >= threshold:
            combinations.append((s, r, idx))

    num_combinations = len(combinations)

    if n_groups > num_combinations:
        raise ValueError(
            f"Cannot form {n_groups} distinct groups from sensitive attributes: "
            f"max is {num_combinations})"
        )

    groups = [idx for (_, _, idx) in combinations]

    # if we need fewer groups, merge the two smallest repeatedly
    while len(groups) > n_groups:
        sizes = [len(g) for g in groups]
        i1, i2 = np.argsort(sizes)[:2]
        # pop the two smallest (remove higher index first)
        grp1 = groups.pop(max(i1, i2))
        grp2 = groups.pop(min(i1, i2))
        groups.append(np.concatenate([grp1, grp2]))

    return groups


def fetch_compas(
    n_groups: int = 2, test_size: float = 0.3, random_state: int = 42
) -> Dataset:
    """
    Fetch and prepare COMPAS (ProPublica ``compas-scores-two-years.csv``).

    Target: ``two_year_recid`` (float in {0.0, 1.0}).
    Groups: intersections of ``sex`` × ``race`` (merged until `n_groups` remain).
    Features are standardized on the training split.

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
        If `n_groups` is larger than the number of available (sex, race) groups.
    """
    # Download the dataset
    df = _download_compas(replace=False)
    # df.dropna(inplace=True)

    X, y, sex, race = _preprocess_compas(df)

    # build stratification labels from numeric codes
    strata = [f"{int(s)}_{int(r)}" for s, r in zip(sex, race)]
    X_train, X_test, y_train, y_test, sex_train, sex_test, race_train, race_test = (
        train_test_split(
            X,
            y,
            sex,
            race,
            test_size=test_size,
            stratify=strata,
            random_state=random_state,
        )
    )

    scaler = StandardScaler()
    scaler.set_output(transform="pandas")
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    group_idx = _partition_sensitive(X_train, sex_train, race_train, n_groups)
    X_train = np.asarray(X_train)
    group_idx_test = _partition_sensitive(X_test, sex_test, race_test, n_groups)
    X_test = np.asarray(X_test)
    return Dataset(
        "COMPAS", X_train, X_test, y_train, y_test, group_idx, group_idx_test
    )
