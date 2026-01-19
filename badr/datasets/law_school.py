from itertools import product
from os import makedirs, path
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from badr.datasets import BADR_PATH, Dataset, fetch_csv


def _download_lawschool(replace: bool = False) -> pd.DataFrame:
    # Download the Law School Admissions Council dataset.
    data_dir = path.join(BADR_PATH, "lsac")
    makedirs(data_dir, exist_ok=True)
    df = fetch_csv(
        url="https://storage.googleapis.com/lawschool_dataset/bar_pass_prediction.csv",
        data_home=data_dir,
        replace=replace,
        sep=",",
        index_col=0,
    )
    return df


def _preprocess_lawschool(
    df,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    # Preprocess the Law School Admissions Council dataset.
    # target
    df = df[
        ["race1", "gender", "age", "fam_inc", "fulltime", "zgpa", "ugpa", "lsat"]
    ].dropna()
    y = df["zgpa"].values

    # Store raw attributes for splitting
    sex = np.asarray(df["gender"].values)  # 'male' or 'female'
    race = np.asarray(df["race1"].values)  # e.g. 'white', 'asian', ...

    # One-hot encode features for modeling
    X = pd.get_dummies(df.iloc[:, :-1], dtype=np.float32)

    X = X.astype(np.float32)

    return X, y, sex, race


def _partition_sensitive(
    X: np.ndarray,
    sex_vals: np.ndarray,
    race_vals: np.ndarray,
    n_groups: int,
    safeguard: bool = True,
) -> List[np.ndarray]:
    # Partition X,y into n_groups by combinations of sex and race categories.
    n_features = X.shape[1]
    sex_cats = np.unique(sex_vals)
    race_cats = np.unique(race_vals)
    all_combinations = list(product(sex_cats, race_cats))

    combinations = []
    for s, r in all_combinations:
        idx = np.where((sex_vals == s) & (race_vals == r))[0]
        threshold = n_features if safeguard else 0
        if idx.size >= threshold:
            combinations.append((s, r, idx))

    num_combinations = len(combinations)

    if n_groups > num_combinations:
        raise ValueError(
            f"Cannot form {n_groups} distinct groups from sensitive attributes: "
            f"max is {num_combinations} (|sex| × |race| = "
            f"{len(sex_cats)}×{len(race_cats)} = {len(all_combinations)} possible, "
            f"{num_combinations} actually present)."
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


def fetch_lawschool(
    n_groups: int = 2, test_size: float = 0.3, random_state: int = 42
) -> Dataset:
    """
    Fetch LSAC (bar pass / law school admissions) dataset.

    Target: ``zgpa``.
    Groups: intersections of ``gender`` × ``race1`` (merged until `n_groups` remain).
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
        If `n_groups` is larger than the number of available (gender, race1) groups.
    """
    # Download the dataset
    df = _download_lawschool(replace=False)
    df.dropna(inplace=True)

    X, y, sex, race = _preprocess_lawschool(df)

    strata = [f"{s.strip()}_{r.strip()}" for s, r in zip(sex, race)]
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
    X_train = X_train.to_numpy()  # type: ignore
    group_idx_test = _partition_sensitive(X_test, sex_test, race_test, n_groups)
    X_test = X_test.to_numpy()
    return Dataset("LSAC", X_train, X_test, y_train, y_test, group_idx, group_idx_test)
