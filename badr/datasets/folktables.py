from os import makedirs, path
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from folktables import ACSDataSource, BasicProblem, state_list  # noqa: E402
from folktables.acs import adult_filter
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler

from badr.datasets import BADR_PATH, Dataset


def _download_folktables(states: Union[str, List[str]], year: int) -> pd.DataFrame:
    # Download the Folktables ACS dataset for given states and year.
    data_dir = path.join(BADR_PATH, "adult")
    makedirs(data_dir, exist_ok=True)
    if year not in [2014, 2015, 2016, 2017, 2018]:
        raise ValueError("Year must be one of [2014, 2015, 2016, 2017, 2018]")

    # If single state, convert to list
    if isinstance(states, str):
        states = [states]

    invalid = [s for s in states if s not in state_list]
    if invalid:
        raise ValueError(
            f"Invalid state(s): {invalid}. Must be included in {state_list}"
        )

    data_dir = path.join(BADR_PATH, "folktables")
    makedirs(data_dir, exist_ok=True)
    data_source = ACSDataSource(
        survey_year=f"{year}", horizon="1-Year", survey="person", root_dir=data_dir
    )
    acs_data = data_source.get_data(states=states, download=True)
    return acs_data


def _download_ACSIncome(
    states: Union[str, List[str]], year: int
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # Download the ACS Income dataset from Folktables.
    acs_data = _download_folktables(states, year)
    common_features = [
        "AGEP",  # age
        "COW",  # class of worker
        "SCHL",  # education
        "MAR",  # marital status
        "OCCP",  # occupation
        "POBP",  # place of birth
        "RELP",  # relationship
        "WKHP",  # hours per week
        "SEX",  # sex
        "RAC1P",  # race
    ]

    def income_transform(x):
        return x > 50_000

    ACSIncome_sex = BasicProblem(
        features=common_features,
        target="PINCP",  # personal income
        target_transform=income_transform,
        group="SEX",
        preprocess=adult_filter,
        postprocess=lambda x: np.nan_to_num(x, nan=-1),
    )
    ACSIncome_race = BasicProblem(
        features=common_features,
        target="PINCP",  # personal income
        target_transform=income_transform,
        group="RAC1P",
        preprocess=adult_filter,
        postprocess=lambda x: np.nan_to_num(x, nan=-1),
    )

    X, y, sex_group = ACSIncome_sex.df_to_pandas(acs_data)
    _, _, race_group = ACSIncome_race.df_to_pandas(acs_data)
    return X, y, sex_group, race_group


def _download_ACSIncomeR(
    states: Union[str, List[str]], year: int
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # Download the ACS Income dataset from Folktables.
    acs_data = _download_folktables(states, year)
    common_features = [
        "AGEP",  # age
        "COW",  # class of worker
        "SCHL",  # education
        "MAR",  # marital status
        "OCCP",  # occupation
        "POBP",  # place of birth
        "RELP",  # relationship
        "WKHP",  # hours per week
        "SEX",  # sex
        "RAC1P",  # race
    ]

    ACSIncome_sex = BasicProblem(
        features=common_features,
        target="PINCP",  # personal income
        group="SEX",
        target_transform=lambda x: np.float32(x),
        preprocess=adult_filter,
        # postprocess=lambda x: np.nan_to_num(x, nan=-1),
    )
    ACSIncome_race = BasicProblem(
        features=common_features,
        target="PINCP",  # personal income
        group="RAC1P",
        target_transform=lambda x: np.float32(x),
        preprocess=adult_filter,
        # postprocess=lambda x: np.nan_to_num(x, nan=-1),
    )

    X, y, sex_group = ACSIncome_sex.df_to_pandas(acs_data)
    _, _, race_group = ACSIncome_race.df_to_pandas(acs_data)
    return X, y, sex_group, race_group


def _download_ACSEmployment(
    states: Union[str, List[str]], year: int
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # Download the ACS Employment dataset from Folktables.
    acs_data = _download_folktables(states, year)
    common_features = [
        "AGEP",
        "SCHL",
        "MAR",
        "RELP",
        "DIS",
        "ESP",
        "CIT",
        "MIG",
        "MIL",
        "ANC",
        "NATIVITY",
        "DEAR",
        "DEYE",
        "DREM",
        "SEX",
        "RAC1P",
    ]

    ACSEmployment_sex = BasicProblem(
        features=common_features,
        target="ESR",  # employment status
        target_transform=lambda x: x == 1,
        group="SEX",
        preprocess=adult_filter,
        postprocess=lambda x: np.nan_to_num(x, nan=-1),
    )
    ACSEmployment_race = BasicProblem(
        features=common_features,
        target="ESR",  # personal income
        target_transform=lambda x: x == 1,
        group="RAC1P",
        preprocess=adult_filter,
        postprocess=lambda x: np.nan_to_num(x, nan=-1),
    )

    X, y, sex_group = ACSEmployment_sex.df_to_pandas(acs_data)
    _, _, race_group = ACSEmployment_race.df_to_pandas(acs_data)
    return X, y, sex_group, race_group


def _download_ACSTravelTime(
    states: Union[str, List[str]], year: int
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # Download the ACS Employment dataset from Folktables.
    def travel_time_filter(data):
        """
        Filters for the employment prediction task
        """
        df = data
        df = df[df["AGEP"] > 16]
        df = df[df["PWGTP"] >= 1]
        df = df[df["ESR"] == 1]
        return df

    acs_data = _download_folktables(states, year)
    common_features = [
        "AGEP",
        "SCHL",
        "MAR",
        "SEX",
        "DIS",
        "ESP",
        "MIG",
        "RELP",
        "RAC1P",
        "PUMA",
        "ST",
        "CIT",
        "OCCP",
        "JWTR",
        "POWPUMA",
        "POVPIP",
    ]

    ACSTravelTimesex = BasicProblem(
        features=common_features,
        target="JWMNP",  # personal income
        target_transform=lambda x: x > 20,
        group="SEX",
        preprocess=travel_time_filter,
        postprocess=lambda x: np.nan_to_num(x, nan=-1),
    )
    ACSTravelTimerace = BasicProblem(
        features=common_features,
        target="JWMNP",  # personal income
        target_transform=lambda x: x > 20,
        group="RAC1P",
        preprocess=travel_time_filter,
        postprocess=lambda x: np.nan_to_num(x, nan=-1),
    )
    ACSTravelTimeage = BasicProblem(
        features=common_features,
        target="JWMNP",  # personal income
        target_transform=lambda x: x > 20,
        group="AGEP",
        preprocess=travel_time_filter,
        postprocess=lambda x: np.nan_to_num(x, nan=-1),
    )

    X, y, sex_group = ACSTravelTimesex.df_to_pandas(acs_data)
    _, _, race_group = ACSTravelTimerace.df_to_pandas(acs_data)
    _, _, age_group = ACSTravelTimeage.df_to_pandas(acs_data)
    return X, y, sex_group, race_group, age_group


def _partition_sensitive(
    X: pd.DataFrame,
    sex_vals: np.ndarray,
    race_vals: np.ndarray,
    n_groups: int = 2,
    age_vals: Union[np.ndarray, None] = None,
) -> List[np.ndarray]:
    # Partition X,y into n_groups by combinations of sex, race, and optionally age categories.
    sex_arr = np.asarray(sex_vals).ravel()
    race_arr = np.asarray(race_vals).ravel()
    # prepare age array if provided
    age_arr: Optional[np.ndarray] = None
    if age_vals is not None:
        age_arr = np.asarray(age_vals).ravel()
        age_arr = np.where(age_arr > 40, 1, 0)

    n_features = X.shape[1]
    buckets: List[np.ndarray] = []
    # build buckets: if age_vals provided, include age in combination
    if age_vals is None:
        # combine sex and race
        for s in np.unique(sex_arr):
            for r in np.unique(race_arr):
                mask = (sex_arr == s) & (race_arr == r)
                pos = np.where(mask)[0]
                if pos.size > n_features:
                    buckets.append(pos)
    else:
        # combine sex, race and age
        assert age_arr is not None, "age_vals provided so age_arr should not be None"
        for s in np.unique(sex_arr):
            for r in np.unique(race_arr):
                for a in np.unique(age_arr):
                    mask = (sex_arr == s) & (race_arr == r) & (age_arr == a)
                    pos = np.where(mask)[0]
                    if pos.size > n_features:
                        buckets.append(pos)

    if len(buckets) < n_groups:
        raise ValueError(f"Cannot form {n_groups} groups; only {len(buckets)} buckets")

    while len(buckets) > n_groups:
        sizes = [b.size for b in buckets]
        i1, i2 = np.argsort(sizes)[:2]
        b1 = buckets.pop(max(i1, i2))
        b2 = buckets.pop(min(i1, i2))
        buckets.append(np.concatenate([b1, b2]))

    # now slice by label and return y as numpy array for indexing
    return buckets


def fetch(
    task_type: str = "income",
    states: List[str] = ["WY"],
    year: int = 2018,
    n_groups: int = 2,
    test_size: float = 0.3,
    random_state: int = 42,
) -> Dataset:
    """
    Fetch a Folktables ACS task and build groups from sensitive attributes.

    `task_type`:
    - ``"income"``: binary (PINCP > 50_000)
    - ``"income_r"``: regression (PINCP)
    - ``"employment"``: binary (ESR == 1)
    - ``"travel_time"``: binary (JWMNP > 20)

    Groups are formed from intersections of ``SEX`` × ``RAC1P`` (and ``AGEP`` for
    ``travel_time``), then merged until `n_groups` remain. Features are
    standardized on the training split.

    Parameters
    ----------
    task_type : {"income", "income_r", "employment", "travel_time"}, default="income"
        Task to load.
    states : list[str], default=["WY"]
        State FIPS codes accepted by folktables.
    year : {2014, 2015, 2016, 2017, 2018}, default=2018
        ACS 1-year survey release year.
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
    NotImplementedError
        If `task_type` is not supported.
    ValueError
        If `year` is invalid or if `n_groups` cannot be formed.
    """
    if task_type == "income":
        X, y, sex, race = _download_ACSIncome(states, year)
        name = "ACSIncome"
    elif task_type == "employment":
        X, y, sex, race = _download_ACSEmployment(states, year)
        name = "ACSEmployment"
    elif task_type == "travel_time":
        X, y, sex, race, age = _download_ACSTravelTime(states, year)
        name = "ACSTravelTime"
    elif task_type == "income_r":
        X, y, sex, race = _download_ACSIncomeR(states, year)
        name = "ACSIncomeR"
    else:
        raise NotImplementedError(
            f"Dataset type {task_type} not implemented. Use 'income', 'employment', or 'travel_time'."
        )
    n_samples = len(y)
    # Stratification BEGIN
    sex_col = sex["SEX"]
    race_col = race["RAC1P"]
    buckets: List[np.ndarray] = []
    n_features = X.shape[1]
    if task_type == "travel_time":
        age_col = age["AGEP"]  # type: ignore
        for s in sex_col.unique():
            for r in race_col.unique():
                for a in age_col.unique():
                    mask = (sex_col == s) & (race_col == r) & (age_col == a)
                    idx = mask[mask].index.to_numpy()
                    if idx.size > n_features:
                        buckets.append(idx)
    else:
        for s in sex_col.unique():
            for r in race_col.unique():
                mask = (sex_col == s) & (race_col == r)
                idx = mask[mask].index.to_numpy()
                if idx.size > n_features:
                    buckets.append(idx)

    if len(buckets) < n_groups:
        raise ValueError(
            f"Cannot form {n_groups} sensitive groups; only {len(buckets)} buckets available"
        )
    # Merge smallest until n_groups
    while len(buckets) > n_groups:
        sizes = [b.shape[0] for b in buckets]
        i1, i2 = np.argsort(sizes)[:2]
        b1 = buckets.pop(max(i1, i2))
        b2 = buckets.pop(min(i1, i2))
        buckets.append(np.concatenate([b1, b2]))

    # Assign stratification labels, then filter to those rows
    strata = np.full(n_samples, -1, dtype=int)
    for gid, b in enumerate(buckets):
        strata[b] = gid
    mask = strata >= 0
    Xf = X.iloc[mask]
    yf = y[mask]
    sexf = sex[mask]
    racef = race[mask]
    strataf = strata[mask]
    # include age column for travel_time
    if task_type == "travel_time":
        agef = age[mask]  # type: ignore
    # Stratification END

    if task_type == "travel_time":
        (
            X_train,
            X_test,
            y_train,
            y_test,
            sex_train,
            sex_test,
            race_train,
            race_test,
            age_train,
            age_test,
        ) = train_test_split(
            Xf,
            yf,
            sexf,
            racef,
            agef,  # type: ignore
            test_size=test_size,
            stratify=strataf,
            random_state=random_state,
        )
    else:
        X_train, X_test, y_train, y_test, sex_train, sex_test, race_train, race_test = (
            train_test_split(
                Xf,
                yf,
                sexf,
                racef,
                test_size=test_size,
                stratify=strataf,
                random_state=random_state,
            )
        )
    y_train = y_train.to_numpy().ravel()
    y_test = y_test.to_numpy().ravel()

    scaler = StandardScaler()
    scaler.set_output(transform="pandas")
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    if task_type == "travel_time":
        group_idx = _partition_sensitive(
            X_train,  # type: ignore
            sex_train,
            race_train,
            n_groups,
            age_train,  # type: ignore
        )  # type: ignore
    else:
        group_idx = _partition_sensitive(X_train, sex_train, race_train, n_groups)  # type: ignore
    X_train = X_train.to_numpy()  # type: ignore
    if task_type == "travel_time":
        group_idx_test = _partition_sensitive(
            X_test,
            sex_test,
            race_test,
            n_groups,
            age_test,  # type: ignore
        )
    else:
        group_idx_test = _partition_sensitive(X_test, sex_test, race_test, n_groups)
    X_test = X_test.to_numpy()
    return Dataset(name, X_train, X_test, y_train, y_test, group_idx, group_idx_test)


def fetch_cv(
    task_type: str = "income",
    states: List[str] = ["WY"],
    year: int = 2018,
    n_groups: int = 2,
    n_splits: int = 5,
    random_state: int = 42,
) -> List[Dataset]:
    """
    K-fold CV version of :func:`~badr.datasets.fetch` for Folktables ACS tasks.

    Uses the same task definitions, stratification buckets, and group construction
    as :func:`~badr.datasets.fetch`. Each fold scales features using training
    statistics for that fold.

    Parameters
    ----------
    task_type : {"income", "income_r", "employment", "travel_time"}, default="income"
        Task to load.
    states : list[str], default=["WY"]
        State FIPS codes accepted by folktables.
    year : {2014, 2015, 2016, 2017, 2018}, default=2018
        ACS 1-year survey release year.
    n_groups : int, default=2
        Number of groups to return.
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
    NotImplementedError
        If `task_type` is not supported.
    ValueError
        If `year` is invalid or if `n_groups` cannot be formed.
    """
    # --- same download/selection logic as in fetch() ---
    if task_type == "income":
        X, y, sex, race = _download_ACSIncome(states, year)
        name = "ACSIncome"
    elif task_type == "employment":
        X, y, sex, race = _download_ACSEmployment(states, year)
        name = "ACSEmployment"
    elif task_type == "travel_time":
        X, y, sex, race, age = _download_ACSTravelTime(states, year)
        name = "ACSTravelTime"
    elif task_type == "income_r":
        X, y, sex, race = _download_ACSIncomeR(states, year)
        name = "ACSIncomeR"
    else:
        raise NotImplementedError(
            f"Dataset type {task_type} not implemented. Use 'income', 'employment', or 'travel_time'."
        )

    n_samples = len(y)

    # --- Stratification BEGIN (copied from fetch) ---
    sex_col = sex["SEX"]
    race_col = race["RAC1P"]
    buckets: List[np.ndarray] = []
    n_features = X.shape[1]

    if task_type == "travel_time":
        age_col = age["AGEP"]  # type: ignore
        for s in sex_col.unique():
            for r in race_col.unique():
                for a in age_col.unique():
                    mask = (sex_col == s) & (race_col == r) & (age_col == a)
                    idx = mask[mask].index.to_numpy()
                    if idx.size > n_features:
                        buckets.append(idx)
    else:
        for s in sex_col.unique():
            for r in race_col.unique():
                mask = (sex_col == s) & (race_col == r)
                idx = mask[mask].index.to_numpy()
                if idx.size > n_features:
                    buckets.append(idx)

    if len(buckets) < n_groups:
        raise ValueError(
            f"Cannot form {n_groups} sensitive groups; only {len(buckets)} buckets available"
        )

    # Merge smallest until n_groups
    while len(buckets) > n_groups:
        sizes = [b.shape[0] for b in buckets]
        i1, i2 = np.argsort(sizes)[:2]
        b1 = buckets.pop(max(i1, i2))
        b2 = buckets.pop(min(i1, i2))
        buckets.append(np.concatenate([b1, b2]))

    # Assign stratification labels, then filter to those rows
    strata = np.full(n_samples, -1, dtype=int)
    for gid, b in enumerate(buckets):
        strata[b] = gid

    mask = strata >= 0
    Xf = X.iloc[mask]
    yf = y[mask]
    sexf = sex[mask]
    racef = race[mask]
    strataf = strata[mask]

    if task_type == "travel_time":
        agef = age[mask]  # type: ignore
    # --- Stratification END ---

    skf = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )

    datasets: List[Dataset] = []

    # K-fold over filtered data
    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(Xf, strataf)):
        # slice per fold
        X_train = Xf.iloc[train_idx].copy()
        X_test = Xf.iloc[test_idx].copy()
        y_train = yf.iloc[train_idx].to_numpy().ravel()
        y_test = yf.iloc[test_idx].to_numpy().ravel()
        sex_train = sexf.iloc[train_idx]
        sex_test = sexf.iloc[test_idx]
        race_train = racef.iloc[train_idx]
        race_test = racef.iloc[test_idx]
        if task_type == "travel_time":
            age_train = agef.iloc[train_idx]  # type: ignore
            age_test = agef.iloc[test_idx]  # type: ignore

        # scale as in fetch()
        scaler = StandardScaler()
        scaler.set_output(transform="pandas")
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # groups as in fetch()
        if task_type == "travel_time":
            group_idx = _partition_sensitive(
                X_train,  # type: ignore
                sex_train,
                race_train,
                n_groups,
                age_train,  # type: ignore
            )
        else:
            group_idx = _partition_sensitive(
                X_train,
                sex_train,
                race_train,
                n_groups,
            )  # type: ignore

        X_train_np = X_train.to_numpy()  # type: ignore

        if task_type == "travel_time":
            group_idx_test = _partition_sensitive(
                X_test,
                sex_test,
                race_test,
                n_groups,
                age_test,  # type: ignore
            )
        else:
            group_idx_test = _partition_sensitive(
                X_test,
                sex_test,
                race_test,
                n_groups,
            )

        X_test_np = X_test.to_numpy()

        group_size_report = ", ".join(
            f"G{gid}: train={len(train_g)} test={len(test_g)}"
            for gid, (train_g, test_g) in enumerate(zip(group_idx, group_idx_test))
        )
        print(f"[fetch_cv] fold {fold_idx}: {group_size_report}")

        datasets.append(
            Dataset(
                f"{name}_fold{fold_idx}",
                X_train_np,
                X_test_np,
                y_train,
                y_test,
                group_idx,
                group_idx_test,
            )
        )

    return datasets


def fetch_ACSIncome(
    states: List[str] = ["WY"],
    year: int = 2018,
    n_groups: int = 2,
    test_size: float = 0.3,
    random_state: int = 42,
) -> Dataset:
    """
    Fetch Folktables ACSIncome (binary income > 50k) and build groups from SEX × RAC1P.

    Parameters
    ----------
    states : list[str], default=["WY"]
        State FIPS codes accepted by folktables.
    year : {2014, 2015, 2016, 2017, 2018}, default=2018
        ACS 1-year survey release year.
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
    """
    return fetch("income", states, year, n_groups, test_size, random_state)


def fetch_ACSIncome_cv(
    states: List[str] = ["WY"],
    year: int = 2018,
    n_groups: int = 2,
    n_splits: int = 5,
    random_state: int = 42,
) -> List[Dataset]:
    """
    K-fold CV version of :func:`~badr.datasets.fetch_ACSIncome`.

    Parameters
    ----------
    states : list[str], default=["WY"]
        State FIPS codes accepted by folktables.
    year : {2014, 2015, 2016, 2017, 2018}, default=2018
        ACS 1-year survey release year.
    n_groups : int, default=2
        Number of groups to return.
    n_splits : int, default=5
        Number of folds.
    random_state : int, default=42
        CV shuffle seed.

    Returns
    -------
    list[Dataset]
        One Dataset per fold.
    """
    return fetch_cv("income", states, year, n_groups, n_splits, random_state)


def fetch_ACSIncomeR(
    states: List[str] = ["WY"],
    year: int = 2018,
    n_groups: int = 2,
    test_size: float = 0.3,
    random_state: int = 42,
) -> Dataset:
    """
    Fetch Folktables ACSIncomeR (PINCP regression) and build groups from SEX × RAC1P.

    Parameters
    ----------
    states : list[str], default=["WY"]
        State FIPS codes accepted by folktables.
    year : {2014, 2015, 2016, 2017, 2018}, default=2018
        ACS 1-year survey release year.
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
    """
    return fetch("income_r", states, year, n_groups, test_size, random_state)


def fetch_ACSEmployment(
    states: List[str] = ["WY"],
    year: int = 2018,
    n_groups: int = 2,
    test_size: float = 0.3,
    random_state: int = 42,
) -> Dataset:
    """
    Fetch Folktables ACSEmployment (ESR == 1) and build groups from SEX × RAC1P.

    Parameters
    ----------
    states : list[str], default=["WY"]
        State FIPS codes accepted by folktables.
    year : {2014, 2015, 2016, 2017, 2018}, default=2018
        ACS 1-year survey release year.
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
    """
    return fetch("employment", states, year, n_groups, test_size, random_state)


def fetch_ACSEmployment_cv(
    states: List[str] = ["WY"],
    year: int = 2018,
    n_groups: int = 2,
    n_splits: int = 5,
    random_state: int = 42,
) -> List[Dataset]:
    """
    K-fold CV version of :func:`~badr.datasets.fetch_ACSEmployment`.

    Parameters
    ----------
    states : list[str], default=["WY"]
        State FIPS codes accepted by folktables.
    year : {2014, 2015, 2016, 2017, 2018}, default=2018
        ACS 1-year survey release year.
    n_groups : int, default=2
        Number of groups to return.
    n_splits : int, default=5
        Number of folds.
    random_state : int, default=42
        CV shuffle seed.

    Returns
    -------
    list[Dataset]
        One Dataset per fold.
    """
    return fetch_cv("employment", states, year, n_groups, n_splits, random_state)


def fetch_ACSTravelTime(
    states: List[str] = ["WY"],
    year: int = 2018,
    n_groups: int = 2,
    test_size: float = 0.3,
    random_state: int = 42,
) -> Dataset:
    """
    Fetch Folktables ACSTravelTime (JWMNP > 20) and build groups from SEX × RAC1P × AGEP.

    Parameters
    ----------
    states : list[str], default=["WY"]
        State FIPS codes accepted by folktables.
    year : {2014, 2015, 2016, 2017, 2018}, default=2018
        ACS 1-year survey release year.
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
    """
    return fetch("travel_time", states, year, n_groups, test_size, random_state)


def fetch_ACSTravelTime_cv(
    states: List[str] = ["WY"],
    year: int = 2018,
    n_groups: int = 2,
    n_splits: int = 5,
    random_state: int = 42,
) -> List[Dataset]:
    """
    K-fold CV version of :func:`~badr.datasets.fetch_ACSTravelTime`.

    Parameters
    ----------
    states : list[str], default=["WY"]
        State FIPS codes accepted by folktables.
    year : {2014, 2015, 2016, 2017, 2018}, default=2018
        ACS 1-year survey release year.
    n_groups : int, default=2
        Number of groups to return.
    n_splits : int, default=5
        Number of folds.
    random_state : int, default=42
        CV shuffle seed.

    Returns
    -------
    list[Dataset]
        One Dataset per fold.
    """
    return fetch_cv("travel_time", states, year, n_groups, n_splits, random_state)
