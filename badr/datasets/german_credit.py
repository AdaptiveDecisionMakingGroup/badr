from os import makedirs, path
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler

from badr.datasets import BADR_PATH, Dataset, fetch_csv

CAT, NUM = "CAT", "NUM"


def _download_germancredit(replace: bool = False) -> pd.DataFrame:
    # Download the German Credit dataset from UCI repository.
    data_dir = path.join(BADR_PATH, "german_credit")
    makedirs(data_dir, exist_ok=True)
    column_names = {
        "Status of Checking Account": CAT,
        "Duration in Month": NUM,
        "Credit History": CAT,
        "Purpose": CAT,
        "Credit Amount": NUM,
        "Savings Account/Bonds": CAT,
        "Employment": CAT,
        "Disposable Income": NUM,
        "Personal Status": CAT,
        "Other Debtors/Guarantors": CAT,
        "Present Residence Since": NUM,
        "Property": CAT,
        "Age": NUM,
        "Other Installment Plans": CAT,
        "Housing": CAT,
        "Number of Existing Credits at This Bank": NUM,
        "Job": CAT,
        "Number of People Being Liable to Provide Maintenance for": NUM,
        "Telephone": CAT,
        "Foreign Worker": CAT,
        "credit": CAT,
    }
    df = fetch_csv(
        url="https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data",
        data_home=data_dir,
        replace=replace,
        header=None,
        sep=" ",
        names=column_names,
    )
    return df


def _preprocess_germancredit(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    # Filter, map target, extract sex, split NUM/CAT columns.
    # initial column type spec
    column_names = {
        "Status of Checking Account": CAT,
        "Duration in Month": NUM,
        "Credit History": CAT,
        "Purpose": CAT,
        "Credit Amount": NUM,
        "Savings Account/Bonds": CAT,
        "Employment": CAT,
        "Disposable Income": NUM,
        "Personal Status": CAT,
        "Other Debtors/Guarantors": CAT,
        "Present Residence Since": NUM,
        "Property": CAT,
        "Age": NUM,
        "Other Installment Plans": CAT,
        "Housing": CAT,
        "Number of Existing Credits at This Bank": NUM,
        "Job": CAT,
        "Number of People Being Liable to Provide Maintenance for": NUM,
        "Telephone": CAT,
        "Foreign Worker": CAT,
        "credit": CAT,
    }
    num_cols = [c for c, t in column_names.items() if t == "NUM"]
    cat_cols = [c for c, t in column_names.items() if t == "CAT" and c != "class"]

    df["single"] = [row in ["A93", "A95"] for row in df["Personal Status"]]
    df["sex"] = [row in ["A92", "A95"] for row in df["Personal Status"]]

    df["single"] = df["single"].astype(float)
    df["sex"] = df["sex"].astype(float)

    df["class"] = df["credit"] - 1

    df.drop(["Personal Status", "credit"], axis=1)

    X_num = df[num_cols].astype(np.float32)
    X_cat = pd.get_dummies(
        df[cat_cols],
        prefix_sep="_",
        dummy_na=False,
        dtype=np.float32,
    )
    X = pd.concat([X_num, X_cat], axis=1)
    X = X.reindex(sorted(X.columns), axis=1)

    y = (df["class"] == 1).astype(float).to_numpy()

    X = pd.get_dummies(df.iloc[:, :-1], dtype=np.float32)
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
        raise ValueError(f"German Credit supports only n_groups=2, got {n_groups}")
    return groups


def fetch_germancredit(
    n_groups: int = 2, test_size: float = 0.3, random_state: int = 42
) -> Dataset:
    """
    Fetch German Credit (UCI Statlog German Credit).

    Target: derived from ``credit`` (binary).
    Groups: ``sex`` extracted from ``Personal Status``. Only `n_groups=2` is supported.
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
    df = _download_germancredit(replace=False)
    X, y, sex = _preprocess_germancredit(df)

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
        "German Credit", X_train, X_test, y_train, y_test, group_idx, group_idx_test
    )


def fetch_germancredit_cv(
    n_groups: int = 2,
    n_splits: int = 5,
    random_state: int = 42,
) -> List[Dataset]:
    """
    Fetch German Credit (UCI Statlog German Credit).

    Target: derived from ``credit`` (binary).
    Groups: ``sex`` extracted from ``Personal Status``. Only `n_groups=2` is supported.
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
    df = _download_germancredit(replace=False)
    X, y, sex = _preprocess_germancredit(df)

    skf = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )

    datasets: List[Dataset] = []

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, sex)):
        # split
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        sex_train, sex_test = sex[train_idx], sex[test_idx]

        # scale (same as in fetch_germancredit)
        scaler = StandardScaler()
        scaler.set_output(transform="pandas")
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # partition by sensitive attribute (sex)
        group_idx = _partition_sensitive(X_train, sex_train, n_groups)
        X_train_np = X_train.to_numpy()
        group_idx_test = _partition_sensitive(X_test, sex_test, n_groups)
        X_test_np = X_test.to_numpy()

        datasets.append(
            Dataset(
                name=f"German Credit fold {fold_idx}",
                X_train=X_train_np,
                X_test=X_test_np,
                y_train=y_train,
                y_test=y_test,
                group_idx=group_idx,
                group_idx_test=group_idx_test,
            )
        )

    return datasets
