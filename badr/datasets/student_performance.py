from os import makedirs, path
from typing import List

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from badr.datasets import BADR_PATH, Dataset


def _download_studentperformance(replace: bool = False) -> pd.DataFrame:
    # Download the Student Performnce dataset via sklearn.fetch_openml.
    data_dir = path.join(BADR_PATH, "student_performance")
    makedirs(data_dir, exist_ok=True)
    ds = fetch_openml(
        name="student-performance-uci",
        version=1,
        as_frame=True,
        data_home=data_dir,
        cache=not replace,
    )
    return ds.frame


def _preprocess_studentperformance(df: pd.DataFrame) -> tuple:
    # Filter, map target, extract sex, split NUM/CAT columns.
    # initial column type spec
    NUM, CAT = "NUM", "CAT"
    column_types = {
        "G3": NUM,
        "sex": CAT,
        "age": NUM,
        "address": CAT,
        "famsize": CAT,
        "Pstatus": CAT,
        "Medu": NUM,
        "Fedu": NUM,
        "Mjob": CAT,
        "Fjob": CAT,
        "reason": CAT,
        "guardian": CAT,
        "traveltime": NUM,
        "studytime": NUM,
        "failures": NUM,
        "schoolsup": CAT,
        "famsup": CAT,
        "paid": CAT,
        "activities": CAT,
        "nursery": CAT,
        "higher": CAT,
        "internet": CAT,
        "romantic": CAT,
        "famrel": NUM,
        "freetime": NUM,
        "goout": NUM,
        "Dalc": NUM,
        "Walc": NUM,
        "health": NUM,
        "absences": NUM,
    }

    y = df["G3"].values
    df["sex"] = df["sex"].map({"F": 1.0, "M": 0.0})
    sex = df["sex"].values

    df = df.drop(columns=["school"], axis=1)
    num_cols = [c for c, t in column_types.items() if t == "NUM"]
    cat_cols = [c for c, t in column_types.items() if t == "CAT" and c != "G3"]

    X_num = df[num_cols].astype(np.float32)
    X_cat = pd.get_dummies(
        df[cat_cols],
        prefix_sep="_",
        dummy_na=False,
        dtype=np.float32,
    )
    X = pd.concat([X_num, X_cat], axis=1)
    X = X.reindex(sorted(X.columns), axis=1)

    return X, y, sex, num_cols


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
        raise ValueError(f"Student Performnce supports only n_groups=2, got {n_groups}")
    return groups


def fetch_studentperformance(
    n_groups: int = 2, test_size: float = 0.3, random_state: int = 42
) -> Dataset:
    """
    Fetch Student Performance (OpenML ``student-performance-uci``, v1).

    Target: ``G3``.
    Groups: ``sex`` (M vs F, mapped to 0.0/1.0). Only `n_groups=2` is supported.
    Numeric features are standardized on the training split.

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
    df = _download_studentperformance(replace=False)
    X, y, sex, num_cols = _preprocess_studentperformance(df)

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
    X_train[num_cols] = scaler.fit_transform(X_train[num_cols].astype(float))
    X_test[num_cols] = scaler.transform(X_test[num_cols].astype(float))

    group_idx = _partition_sensitive(X_train, sex_train, n_groups)
    X_train = X_train.to_numpy()  # type: ignore
    group_idx_test = _partition_sensitive(X_test, sex_test, n_groups)
    X_test = X_test.to_numpy()

    return Dataset(
        "Student Performance",
        X_train,
        X_test,
        y_train,
        y_test,
        group_idx,
        group_idx_test,
    )
