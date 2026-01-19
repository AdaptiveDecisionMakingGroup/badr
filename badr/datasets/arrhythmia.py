from os import makedirs, path
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler

from badr.datasets import BADR_PATH, Dataset


def _download_arrhythmia(replace: bool = False) -> pd.DataFrame:
    # Download the Arrhythmia dataset via sklearn.fetch_openml.
    data_dir = path.join(BADR_PATH, "arrhythmia")
    makedirs(data_dir, exist_ok=True)
    ds = fetch_openml(
        name="arrhythmia",
        version=2,
        as_frame=True,
        data_home=data_dir,
        cache=not replace,
    )
    return ds.frame


def _preprocess_arrhythmia(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, List[str]]:
    # Filter, map target, extract sex, split NUM/CAT columns.
    # initial column type spec
    CAT, NUM = "CAT", "NUM"
    column_types = {
        "binaryClass": CAT,
        "age": NUM,
        "sex": CAT,
        "height": NUM,
        "weight": NUM,
        "QRSduration": NUM,
        "PRinterval": NUM,
        "Q-Tinterval": NUM,
        "Tinterval": NUM,
        "Pinterval": NUM,
        "QRS": NUM,
        "T": NUM,
        "P": NUM,
        "J": NUM,
        "heartrate": NUM,
        "chDI_Qwave": NUM,
        "chDI_Rwave": NUM,
        "chDI_Swave": NUM,
        "chDI_RPwave": NUM,
        "chDI_SPwave": NUM,
        "chDI_intrinsicReflecttions": NUM,
        "chDII_Qwave": NUM,
        "chDII_Rwave": NUM,
        "chDII_Swave": NUM,
        "chDII_RPwave": NUM,
        "chDII_SPwave": NUM,
        "chDII_intrinsicReflecttions": NUM,
        "chDIII_Qwave": NUM,
        "chDIII_Rwave": NUM,
        "chDIII_Swave": NUM,
        "chDIII_RPwave": NUM,
        "chDIII_SPwave": NUM,
        "chDIII_intrinsicReflecttions": NUM,
        "chAVR_Qwave": NUM,
        "chAVR_Rwave": NUM,
        "chAVR_Swave": NUM,
        "chAVR_RPwave": NUM,
        "chAVR_SPwave": NUM,
        "chAVR_intrinsicReflecttions": NUM,
        "chAVL_Qwave": NUM,
        "chAVL_Rwave": NUM,
        "chAVL_Swave": NUM,
        "chAVL_RPwave": NUM,
        "chAVL_SPwave": NUM,
        "chAVL_intrinsicReflecttions": NUM,
        "chAVF_Qwave": NUM,
        "chAVF_Rwave": NUM,
        "chAVF_Swave": NUM,
        "chAVF_RPwave": NUM,
        "chAVF_SPwave": NUM,
        "chAVF_intrinsicReflecttions": NUM,
        "chV1_Qwave": NUM,
        "chV1_Rwave": NUM,
        "chV1_Swave": NUM,
        "chV1_RPwave": NUM,
        "chV1_SPwave": NUM,
        "chV1_intrinsicReflecttions": NUM,
        "chV2_Qwave": NUM,
        "chV2_Rwave": NUM,
        "chV2_Swave": NUM,
        "chV2_RPwave": NUM,
        "chV2_SPwave": NUM,
        "chV2_intrinsicReflecttions": NUM,
        "chV3_Qwave": NUM,
        "chV3_Rwave": NUM,
        "chV3_Swave": NUM,
        "chV3_RPwave": NUM,
        "chV3_SPwave": NUM,
        "chV3_intrinsicReflecttions": NUM,
        "chV4_Qwave": NUM,
        "chV4_Rwave": NUM,
        "chV4_Swave": NUM,
        "chV4_RPwave": NUM,
        "chV4_SPwave": NUM,
        "chV4_intrinsicReflecttions": NUM,
        "chV5_Qwave": NUM,
        "chV5_Rwave": NUM,
        "chV5_Swave": NUM,
        "chV5_RPwave": NUM,
        "chV5_SPwave": NUM,
        "chV5_intrinsicReflecttions": NUM,
        "chV6_Qwave": NUM,
        "chV6_Rwave": NUM,
        "chV6_Swave": NUM,
        "chV6_RPwave": NUM,
        "chV6_SPwave": NUM,
        "chV6_intrinsicReflecttions": NUM,
        "chDI_JJwaveAmp": NUM,
        "chDI_QwaveAmp": NUM,
        "chDI_RwaveAmp": NUM,
        "chDI_SwaveAmp": NUM,
        "chDI_RPwaveAmp": NUM,
        "chDI_SPwaveAmp": NUM,
        "chDI_PwaveAmp": NUM,
        "chDI_TwaveAmp": NUM,
        "chDI_QRSA": NUM,
        "chDI_QRSTA": NUM,
        "chDII_JJwaveAmp": NUM,
        "chDII_QwaveAmp": NUM,
        "chDII_RwaveAmp": NUM,
        "chDII_SwaveAmp": NUM,
        "chDII_RPwaveAmp": NUM,
        "chDII_SPwaveAmp": NUM,
        "chDII_PwaveAmp": NUM,
        "chDII_TwaveAmp": NUM,
        "chDII_QRSA": NUM,
        "chDII_QRSTA": NUM,
        "chDIII_JJwaveAmp": NUM,
        "chDIII_QwaveAmp": NUM,
        "chDIII_RwaveAmp": NUM,
        "chDIII_SwaveAmp": NUM,
        "chDIII_RPwaveAmp": NUM,
        "chDIII_SPwaveAmp": NUM,
        "chDIII_PwaveAmp": NUM,
        "chDIII_TwaveAmp": NUM,
        "chDIII_QRSA": NUM,
        "chDIII_QRSTA": NUM,
        "chAVR_JJwaveAmp": NUM,
        "chAVR_QwaveAmp": NUM,
        "chAVR_RwaveAmp": NUM,
        "chAVR_SwaveAmp": NUM,
        "chAVR_RPwaveAmp": NUM,
        "chAVR_SPwaveAmp": NUM,
        "chAVR_PwaveAmp": NUM,
        "chAVR_TwaveAmp": NUM,
        "chAVR_QRSA": NUM,
        "chAVR_QRSTA": NUM,
        "chAVL_JJwaveAmp": NUM,
        "chAVL_QwaveAmp": NUM,
        "chAVL_RwaveAmp": NUM,
        "chAVL_SwaveAmp": NUM,
        "chAVL_RPwaveAmp": NUM,
        "chAVL_SPwaveAmp": NUM,
        "chAVL_PwaveAmp": NUM,
        "chAVL_TwaveAmp": NUM,
        "chAVL_QRSA": NUM,
        "chAVL_QRSTA": NUM,
        "chAVF_JJwaveAmp": NUM,
        "chAVF_QwaveAmp": NUM,
        "chAVF_RwaveAmp": NUM,
        "chAVF_SwaveAmp": NUM,
        "chAVF_RPwaveAmp": NUM,
        "chAVF_SPwaveAmp": NUM,
        "chAVF_PwaveAmp": NUM,
        "chAVF_TwaveAmp": NUM,
        "chAVF_QRSA": NUM,
        "chAVF_QRSTA": NUM,
        "chV1_JJwaveAmp": NUM,
        "chV1_QwaveAmp": NUM,
        "chV1_RwaveAmp": NUM,
        "chV1_SwaveAmp": NUM,
        "chV1_RPwaveAmp": NUM,
        "chV1_SPwaveAmp": NUM,
        "chV1_PwaveAmp": NUM,
        "chV1_TwaveAmp": NUM,
        "chV1_QRSA": NUM,
        "chV1_QRSTA": NUM,
        "chV2_JJwaveAmp": NUM,
        "chV2_QwaveAmp": NUM,
        "chV2_RwaveAmp": NUM,
        "chV2_SwaveAmp": NUM,
        "chV2_RPwaveAmp": NUM,
        "chV2_SPwaveAmp": NUM,
        "chV2_PwaveAmp": NUM,
        "chV2_TwaveAmp": NUM,
        "chV2_QRSA": NUM,
        "chV2_QRSTA": NUM,
        "chV3_JJwaveAmp": NUM,
        "chV3_QwaveAmp": NUM,
        "chV3_RwaveAmp": NUM,
        "chV3_SwaveAmp": NUM,
        "chV3_RPwaveAmp": NUM,
        "chV3_SPwaveAmp": NUM,
        "chV3_PwaveAmp": NUM,
        "chV3_TwaveAmp": NUM,
        "chV3_QRSA": NUM,
        "chV3_QRSTA": NUM,
        "chV4_JJwaveAmp": NUM,
        "chV4_QwaveAmp": NUM,
        "chV4_RwaveAmp": NUM,
        "chV4_SwaveAmp": NUM,
        "chV4_RPwaveAmp": NUM,
        "chV4_SPwaveAmp": NUM,
        "chV4_PwaveAmp": NUM,
        "chV4_TwaveAmp": NUM,
        "chV4_QRSA": NUM,
        "chV4_QRSTA": NUM,
        "chV5_JJwaveAmp": NUM,
        "chV5_QwaveAmp": NUM,
        "chV5_RwaveAmp": NUM,
        "chV5_SwaveAmp": NUM,
        "chV5_RPwaveAmp": NUM,
        "chV5_SPwaveAmp": NUM,
        "chV5_PwaveAmp": NUM,
        "chV5_TwaveAmp": NUM,
        "chV5_QRSA": NUM,
        "chV5_QRSTA": NUM,
        "chV6_JJwaveAmp": NUM,
        "chV6_QwaveAmp": NUM,
        "chV6_RwaveAmp": NUM,
        "chV6_SwaveAmp": NUM,
        "chV6_RPwaveAmp": NUM,
        "chV6_SPwaveAmp": NUM,
        "chV6_PwaveAmp": NUM,
        "chV6_TwaveAmp": NUM,
        "chV6_QRSA": NUM,
        "chV6_QRSTA": NUM,
    }
    # filter out unrealistic heights
    df = df[df["height"] < 500]

    # select remaining columns and clean missing
    cols = list(column_types.keys())
    df = df[cols]
    df = df.replace("?", np.nan).dropna(axis=0, how="any").reset_index(drop=True)

    # map target to float
    y = df["binaryClass"].map({"N": 0.0, "P": 1.0}).astype(np.float32).to_numpy()

    # extract sensitive attribute and drop it from features
    original_sex = df["sex"].astype(int).to_numpy()
    X = df.drop(columns=["binaryClass", "sex"])

    # split NUM vs CAT for scaling
    num_cols = [c for c, t in column_types.items() if t == NUM and c != "binaryClass"]
    # note: 'sex' is CAT so not scaled

    return X, y, original_sex, num_cols


def _partition_sensitive(
    X: pd.DataFrame, sex_vals: np.ndarray, n_groups: int
) -> List[np.ndarray]:
    # Partition X, y into two groups by sex flag only.
    idx = np.arange(len(X))
    groups = [
        idx[sex_vals == 0],
        idx[sex_vals == 1],
    ]
    if n_groups != 2:
        raise ValueError(f"Arrhythmia supports only n_groups=2, got {n_groups}")
    return groups


def fetch_arrhythmia(
    n_groups: int = 2, test_size: float = 0.3, random_state: int = 42
) -> Dataset:
    """
    Fetch Arrhythmia (OpenML ``arrhythmia``, v2).

    Target: ``binaryClass`` -> {``N``: 0.0, ``P``: 1.0}.
    Groups: ``sex`` (0 vs 1). Only `n_groups=2` is supported.

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
    df = _download_arrhythmia(replace=False)
    X, y, sex, num_cols = _preprocess_arrhythmia(df)

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
    X_train = X_train.to_numpy()
    group_idx_test = _partition_sensitive(X_test, sex_test, n_groups)
    X_test = X_test.to_numpy()

    return Dataset(
        "Arrhythmia", X_train, X_test, y_train, y_test, group_idx, group_idx_test
    )


def fetch_arrhythmia_cv(
    n_groups: int = 2,
    n_splits: int = 5,
    random_state: int = 42,
) -> List[Dataset]:
    """
    Fetch Arrhythmia as K-fold CV splits.

    Same preprocessing as :func:`~badr.datasets.fetch_arrhythmia`. Groups are
    defined by ``sex`` (0 vs 1), so only `n_groups=2` is supported.

    Parameters
    ----------
    n_groups : int, default=2
        Must be 2.
    n_splits : int, default=5
        Number of folds.
    random_state : int, default=42
        CV shuffle seed.

    Returns
    -------
    list[Dataset]
        One Dataset per fold.

    Raises
    ------
    ValueError
        If `n_groups` is not 2.
    """
    df = _download_arrhythmia(replace=False)
    X, y, sex, num_cols = _preprocess_arrhythmia(df)

    skf = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )

    datasets: List[Dataset] = []

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, sex)):
        # slice per fold
        X_train, X_test = X.iloc[train_idx].copy(), X.iloc[test_idx].copy()
        y_train, y_test = y[train_idx], y[test_idx]
        sex_train, sex_test = sex[train_idx], sex[test_idx]

        # scale numeric cols on train, apply to test (same as in fetch_arrhythmia)
        scaler = StandardScaler()
        X_train[num_cols] = scaler.fit_transform(X_train[num_cols].astype(float))
        X_test[num_cols] = scaler.transform(X_test[num_cols].astype(float))

        # partition by sensitive attribute (sex)
        group_idx = _partition_sensitive(X_train, sex_train, n_groups)
        X_train_np = X_train.to_numpy()
        group_idx_test = _partition_sensitive(X_test, sex_test, n_groups)
        X_test_np = X_test.to_numpy()

        datasets.append(
            Dataset(
                name=f"Arrhythmia fold {fold_idx}",
                X_train=X_train_np,
                X_test=X_test_np,
                y_train=y_train,
                y_test=y_test,
                group_idx=group_idx,
                group_idx_test=group_idx_test,
            )
        )

    return datasets
