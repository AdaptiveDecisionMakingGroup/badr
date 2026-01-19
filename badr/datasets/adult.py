from itertools import product
from os import makedirs, path
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from badr.datasets import BADR_PATH, Dataset


def _download_adult(replace: bool = False) -> pd.DataFrame:
    # Download the Adult dataset from OpenML.
    data_dir = path.join(BADR_PATH, "adult")
    makedirs(data_dir, exist_ok=True)
    ds = fetch_openml(
        name="adult",
        version=2,
        as_frame=True,
        data_home=data_dir,
        cache=not replace,
    )
    df = ds.frame

    return df


def _preprocess_adult(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    # Preprocess the Adult dataset.
    # target
    y = df["class"].map({"<=50K": 0.0, ">50K": 1.0}).astype(np.float32).to_numpy()

    # sensitive attributes
    sex = df["sex"].to_numpy()
    race = df["race"].to_numpy()
    column_types = {
        "age": "NUM",
        "workclass": "CAT",
        "fnlwgt": "NUM",
        "education": "CAT",
        "education-num": "NUM",
        "marital-status": "CAT",
        "occupation": "CAT",
        "relationship": "CAT",
        "race": "CAT",
        "sex": "CAT",
        "capital-gain": "NUM",
        "capital-loss": "NUM",
        "hours-per-week": "NUM",
        "native-country": "CAT",
        "class": "CAT",
    }

    num_cols = [c for c, t in column_types.items() if t == "NUM"]
    cat_cols = [c for c, t in column_types.items() if t == "CAT" and c != "class"]

    X_num = df[num_cols].astype(np.float32)
    X_cat = pd.get_dummies(
        df[cat_cols],
        prefix_sep="_",
        dummy_na=False,
        dtype=np.float32,
    )
    X = pd.concat([X_num, X_cat], axis=1)
    X = X.reindex(sorted(X.columns), axis=1)

    return X, y, sex, race, num_cols


def _partition_sensitive(
    X: pd.DataFrame,
    sex_vals: np.ndarray,
    race_vals: np.ndarray,
    n_groups: int,
) -> List[np.ndarray]:
    # Partition the dataset into groups based on sensitive attributes.
    n_features = X.shape[1]
    sex_cats = np.unique(sex_vals)
    race_cats = np.unique(race_vals)
    all_combinations = list(product(sex_cats, race_cats))
    combinations = []
    for s, r in all_combinations:
        idx = np.where((sex_vals == s) & (race_vals == r))[0]
        threshold = n_features
        if idx.size > threshold:
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


def fetch_adult(
    n_groups: int = 2, test_size: float = 0.3, random_state: int = 42
) -> Dataset:
    """
    Fetch and prepare the Adult dataset (OpenML: ``adult``, v2).

    Target: ``class`` mapped to {``<=50K``: 0.0, ``>50K``: 1.0}.
    Groups: intersections of ``sex`` × ``race`` (merged until `n_groups` remain).
    Numeric features are standardized on the training split.

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
    df = _download_adult(replace=False)
    df.dropna(inplace=True)

    X, y, sex, race, num_cols = _preprocess_adult(df)

    # STRATIFICATION BEGIN
    n_samples, n_features = X.shape[0], X.shape[1]
    # 1) build all (sex, race) buckets that are big enough
    buckets = []
    for s in np.unique(sex):
        for r in np.unique(race):
            idx = np.where((sex == s) & (race == r))[0]
            if idx.size > n_features:
                buckets.append(idx)

    # 2) merge the two smallest until you have exactly n_groups
    while len(buckets) > n_groups:
        sizes = [len(b) for b in buckets]
        i1, i2 = np.argsort(sizes)[:2]
        b_small = buckets.pop(max(i1, i2))
        b_next = buckets.pop(min(i1, i2))
        buckets.append(np.concatenate([b_small, b_next]))

    # 3) assign each row to its group‐ID (or −1 if dropped)
    strata = np.full(n_samples, -1, dtype=int)
    for gid, idx in enumerate(buckets):
        strata[idx] = gid

    # 4) drop any rows with strata<0 (those that couldn’t form a bucket)
    mask = strata >= 0
    Xf, yf, strataf = X.iloc[mask], y[mask], strata[mask]
    # STRATIFICATION END

    strata = [f"{s.strip()}_{r.strip()}" for s, r in zip(sex, race)]
    X_train, X_test, y_train, y_test, sex_train, sex_test, race_train, race_test = (
        train_test_split(
            Xf,
            yf,
            sex,
            race,
            test_size=test_size,
            stratify=strataf,
            random_state=random_state,
        )
    )

    scaler = StandardScaler()
    X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
    X_test[num_cols] = scaler.transform(X_test[num_cols])

    group_idx = _partition_sensitive(X_train, sex_train, race_train, n_groups)
    X_train = X_train.to_numpy()
    group_idx_test = _partition_sensitive(X_test, sex_test, race_test, n_groups)
    X_test = X_test.to_numpy()

    return Dataset("Adult", X_train, X_test, y_train, y_test, group_idx, group_idx_test)
