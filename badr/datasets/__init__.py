import os
from os.path import basename, exists, join
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import urlretrieve

import numpy as np
import pandas as pd

BADR_PATH = str(Path.home()) + "/badr_data/"


def fetch_csv(
    url: str,
    data_home: str,
    filename: str = "",
    replace: bool = False,
    **read_csv_kwargs,
) -> pd.DataFrame:
    """
    Download a CSV file if needed and return it as a DataFrame.

    Parameters
    ----------
    url : str
        CSV URL.
    data_home : str
        Directory where the file is cached.
    filename : str, default=""
        Local filename. If empty, uses the basename of `url`.
    replace : bool, default=False
        If True, re-download even if the file already exists.
    **read_csv_kwargs
        Passed to `pandas.read_csv`.

    Returns
    -------
    pandas.DataFrame
        Loaded CSV.
    """
    os.makedirs(data_home, exist_ok=True)
    if filename == "":
        filename = basename(urlparse(url).path)
    filepath = join(data_home, filename)
    if replace or not exists(filepath):
        print(f"Downloading {url!r} to {filepath!r}...")
        urlretrieve(url, filepath)
    return pd.read_csv(filepath, **read_csv_kwargs)


class Dataset:
    """
    Simple container for train/test arrays and per-group indices.
    """

    def __init__(
        self,
        name: str,
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray,
        group_idx: list,
        group_idx_test: list,
    ):
        """
        Parameters
        ----------
        name : str
            Dataset name.
        X_train, X_test : numpy.ndarray
            Feature matrices.
        y_train, y_test : numpy.ndarray
            Targets.
        group_idx, group_idx_test : list
            Per-group selectors for train/test. Each entry is either a boolean mask
            over the corresponding split or an array/list of integer indices.
        """
        self.name = name
        self.X_train = X_train
        self.y_train = y_train
        self.X_train_list = [X_train[g] for g in group_idx]
        self.y_train_list = [y_train[g] for g in group_idx]
        self.X_test_list = [X_test[g] for g in group_idx_test]
        self.y_test_list = [y_test[g] for g in group_idx_test]
        self.groups = group_idx
        self.groups_test = group_idx_test
        self.X_test = X_test
        self.y_test = y_test
        self.n_groups = len((group_idx))
        self.n_features = X_train.shape[1]
        self.dset_train = X_train, y_train
        self.dset_test = X_test, y_test

    def subsample(self, n: int, random_state: int | None = 42) -> "Dataset":
        """
        Subsample the training split and rebuild training group indices.

        Parameters
        ----------
        n : int
            Number of training samples to keep.
        random_state : int | None, default=42
            Seed for sampling. If None, uses an unseeded generator.

        Returns
        -------
        Dataset
            New dataset with subsampled training data and updated `groups`.

        Raises
        ------
        ValueError
            If `n` is not in [1, n_train], or if a boolean group mask has the wrong length.
        """
        n_total = self.X_train.shape[0]
        if n <= 0:
            raise ValueError("n must be a positive integer.")
        if n > n_total:
            raise ValueError(
                f"Requested n={n}, but training set has only {n_total} samples."
            )

        rng = np.random.default_rng(random_state)

        # Choose indices in the ORIGINAL training set
        chosen_idx = rng.choice(n_total, size=n, replace=False)

        # Sort to have a stable mapping original -> new
        chosen_idx = np.sort(chosen_idx)

        # Subsampled training data
        X_train_sub = self.X_train[chosen_idx]
        y_train_sub = self.y_train[chosen_idx]

        # Map original index -> index in subsampled training set
        orig_to_new = {orig: new for new, orig in enumerate(chosen_idx)}

        # Rebuild group_idx for the new training set
        new_group_idx = []
        for g in self.groups:
            g_arr = np.asarray(g)

            # Case 1: g is a boolean mask over the original training set
            if g_arr.dtype == bool:
                if g_arr.shape[0] != n_total:
                    raise ValueError(
                        "Boolean group mask length does not match training set."
                    )
                # New mask is just the old mask restricted to chosen_idx
                new_g = g_arr[chosen_idx]  # boolean mask of length n

            # Case 2: g is an array/list of integer indices
            else:
                new_g = np.array(
                    [orig_to_new[int(i)] for i in g_arr if int(i) in orig_to_new],
                    dtype=int,
                )

            new_group_idx.append(new_g)

        # Build and return a NEW Dataset instance
        return Dataset(
            name=f"{self.name}_sub{n}",
            X_train=X_train_sub,
            X_test=self.X_test,
            y_train=y_train_sub,
            y_test=self.y_test,
            group_idx=new_group_idx,
            group_idx_test=self.groups_test,
        )


from .adult import fetch_adult  # noqa: E402
from .arrhythmia import fetch_arrhythmia, fetch_arrhythmia_cv  # noqa: E402
from .communities_and_crime import fetch_communities_and_crime  # noqa: E402
from .compas import fetch_compas  # noqa: E402
from .dataframe import load_dataframe  # noqa: E402
from .folktables import (  # noqa: E402
    fetch_ACSEmployment,
    fetch_ACSIncome,
    fetch_ACSIncomeR,
    fetch_ACSTravelTime,
    fetch_ACSEmployment_cv,
    fetch_ACSIncome_cv,
    fetch_ACSTravelTime_cv,
)
from .german_credit import fetch_germancredit, fetch_germancredit_cv  # noqa: E402
from .law_school import fetch_lawschool  # noqa: E402
from .parkinsons_telemonitoring import fetch_parkinsons  # noqa: E402
from .student_performance import fetch_studentperformance  # noqa: E402

__all__ = [
    "fetch_ACSIncome",
    "fetch_ACSIncomeR",
    "fetch_ACSEmployment",
    "fetch_ACSTravelTime",
    "fetch_adult",
    "fetch_arrhythmia",
    "fetch_communities_and_crime",
    "fetch_compas",
    "fetch_germancredit",
    "fetch_lawschool",
    "fetch_parkinsons",
    "fetch_studentperformance",
    "load_dataframe",
]
