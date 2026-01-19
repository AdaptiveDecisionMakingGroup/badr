"""
In this experiment, we evaluate the accuracy–fairness trade-off of five Pareto-optimal strategies on classification and regression tasks.

Classification setting:
- Datasets: 51 random splits of ACSEmployment and 51 random splits of ACSTravelTime (102 datasets total), 70/30 train–test, gender partition (2 groups).
- Model/metric: Logistic Regression with $\ell_2^2 = 10^{-2}$ and Individual Fairness.
- Report: mean and standard deviation of test accuracy and test unfairness across the 102 datasets.

Regression setting:
- Datasets: Law School, Parkinson's Telemonitoring, Communities and Crime, Student Performance, each with 10 random 70/30 train–test splits (gender partition when available).
- Model/metric: Ridge Regression with $\ell_2^2 = 10^{-1}$, RMSE as the performance metric, and Demographic Parity for fairness.
- Report: mean and standard deviation of test RMSE and test unfairness across the 40 runs (4 datasets × 10 splits).
"""

from __future__ import annotations

import gc
import json
import os
import sys
from typing import Callable, Iterable, Iterator, List, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import badr

state_list = [
    "AL",
    "AK",
    "AR",
    "CO",
    "CT",
    "DE",
    "HI",
    "ID",
    "IN",
    "IA",
    "KS",
    "KY",
    "LA",
    "ME",
    "MD",
    "MN",
    "MS",
    "MO",
    "MT",
    "NE",
    "NV",
    "NH",
    "NM",
    "ND",
    "OK",
    "OR",
    "RI",
    "SC",
    "SD",
    "UT",
    "VT",
    "VA",
    "WA",
    "WV",
    "WI",
    "WY",
    "PR",
]


def accuracy_and_fairness(
    model,
    metric,
    dset,
    weights: jnp.ndarray,
    task: str,
) -> dict:
    """Fit model with given weights and return performance + fairness."""

    metric.set_model(model)
    model.set_group_weights(jnp.asarray(weights, dtype=jnp.float64))
    model.fit(dset.X_train, dset.y_train, dset.groups)

    if task == "classification":
        train_score = float(model.score(dset.X_train, dset.y_train))
        test_score = float(model.score(dset.X_test, dset.y_test))
        score_keys = ("train_acc", "test_acc")
    elif task == "regression":
        y_pred_train = model.predict(dset.X_train)
        y_pred_test = model.predict(dset.X_test)
        train_score = float(np.sqrt(np.mean((y_pred_train - dset.y_train) ** 2)))
        test_score = float(np.sqrt(np.mean((y_pred_test - dset.y_test) ** 2)))
        score_keys = ("train_rmse", "test_rmse")
    else:
        raise ValueError("task must be 'classification' or 'regression'")

    train_metric = float(metric.fun(model.coef_, dset, train_test="train").item())
    test_metric = float(metric.fun(model.coef_, dset, train_test="test").item())

    return {
        score_keys[0]: train_score,
        score_keys[1]: test_score,
        "train_metric": train_metric,
        "test_metric": test_metric,
        "weights": np.asarray(weights, dtype=float),
    }


def balanced_weights(dset) -> jnp.ndarray:
    sizes = [X.shape[0] for X in dset.X_train_list]
    k = len(sizes)
    if k < 2:
        raise ValueError("balanced_weights requires at least 2 groups")
    n = sum(sizes)
    weights = [(n - s) / ((k - 1) * n) for s in sizes]
    return jnp.asarray(weights, dtype=jnp.float64)


def one_fit_search(model_factory, metric_factory, dset, task: str) -> dict:
    k = len(dset.X_train_list)
    best = None
    for gid in range(k):
        w = jnp.zeros(k, dtype=jnp.float64).at[gid].set(1.0)
        res = accuracy_and_fairness(model_factory(), metric_factory(), dset, w, task)
        if best is None or res["train_metric"] < best["train_metric"]:
            best = res
    return best


def badr_search(
    model_factory,
    metric_factory,
    dset,
    task: str,
    n_samples: int = 200,
    seed: int = 0,
) -> dict:
    k = len(dset.X_train_list)
    rng = np.random.default_rng(seed)
    best = None
    for w in rng.dirichlet(np.ones(k), size=n_samples):
        res = accuracy_and_fairness(model_factory(), metric_factory(), dset, w, task)
        if best is None or res["train_metric"] < best["train_metric"]:
            best = res
    return best


def minmax_eval(model_factory, metric_factory, dset, task: str) -> dict:
    base_model = model_factory()
    metric = metric_factory()

    if isinstance(base_model, badr.models.LogisticRegression):
        mm_model = badr.models.NonsmoothMinMaxLogisticRegression(
            l2_reg=base_model.l2_reg
        )
        mm_model.fit(dset.X_train_list, dset.y_train_list)
    elif isinstance(base_model, badr.models.RidgeRegression):
        mm_model = badr.models.NSMMRR(l2_reg=base_model.l2_reg)
        mm_model.fit(dset.X_train_list, dset.y_train_list)
    else:
        raise ValueError(
            "MinMax is supported for LogisticRegression and RidgeRegression only"
        )

    gw = jnp.asarray(mm_model.group_weights_, dtype=jnp.float64)
    return accuracy_and_fairness(model_factory(), metric, dset, gw, task)


def evaluate_dataset(
    dset_name: str,
    dset,
    task: str,
    model_factory: Callable[[], object],
    metric_factory: Callable[[], object],
    badr_samples: int = 200,
    seed: int = 0,
) -> List[dict]:
    k = len(dset.X_train_list)
    uniform_w = jnp.full((k,), 1.0 / k, dtype=jnp.float64)
    methods: List[Tuple[str, Callable[[], dict]]] = [
        (
            "ERM",
            lambda: accuracy_and_fairness(
                model_factory(), metric_factory(), dset, uniform_w, task
            ),
        ),
        (
            "Balanced",
            lambda: accuracy_and_fairness(
                model_factory(), metric_factory(), dset, balanced_weights(dset), task
            ),
        ),
        ("One-Fit", lambda: one_fit_search(model_factory, metric_factory, dset, task)),
        (
            "BADR",
            lambda: badr_search(
                model_factory,
                metric_factory,
                dset,
                task,
                n_samples=badr_samples,
                seed=seed,
            ),
        ),
        ("MinMax", lambda: minmax_eval(model_factory, metric_factory, dset, task)),
    ]

    records = []
    for method_name, fn in methods:
        res = fn()
        res["method"] = method_name
        res["dset"] = dset_name
        records.append(res)
    return records


def summarize(df: pd.DataFrame, score_col: str) -> pd.DataFrame:
    return (
        df.groupby("method")[[score_col, "test_metric"]]
        .agg(["mean", "std"])
        .sort_index()
    )


def append_records(path: str, records: List[dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)

    def _jsonable(x):
        if isinstance(x, (np.ndarray, jnp.ndarray)):
            return x.tolist()
        if isinstance(x, (np.generic,)):
            return x.item()
        return x

    with open(path, "a", encoding="utf-8") as f:
        for rec in records:
            safe_rec = {k: _jsonable(v) for k, v in rec.items()}
            f.write(json.dumps(safe_rec) + "\n")


def load_completed(path: str, expected_methods: int) -> set:
    completed = set()
    if not os.path.exists(path):
        return completed
    counts = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            dset = rec.get("dset")
            if dset is None:
                continue
            counts[dset] = counts.get(dset, 0) + 1
            if counts[dset] >= expected_methods:
                completed.add(dset)
    return completed


def print_instance_table(records: List[dict], task: str, dset_name: str) -> None:
    if not records:
        return
    df = pd.DataFrame.from_records(records)
    if task == "classification":
        score_col, score_label = "test_acc", "ACC"
    else:
        score_col, score_label = "test_rmse", "RMSE"
    table = (
        df.groupby("method")[[score_col, "test_metric"]]
        .mean()
        .rename(columns={score_col: score_label, "test_metric": "Fairness"})
        .round(4)
    )
    print(f"\n[{dset_name}] mean {score_label} and fairness by method:")
    print(table.to_string())


def iter_classification_dsets() -> Iterator[Tuple[str, object, int]]:
    # One split per state (51 states) for each dataset
    for idx, st in enumerate(state_list):
        seed = idx  # deterministic seed per state
        yield (
            f"ACSEmployment_{st}",
            badr.datasets.fetch_ACSEmployment(
                states=[st], year=2018, n_groups=2, test_size=0.3, random_state=seed
            ),
            seed,
        )
    for idx, st in enumerate(state_list):
        seed = idx  # deterministic seed per state
        yield (
            f"ACSTravelTime_{st}",
            badr.datasets.fetch_ACSTravelTime(
                states=[st], year=2018, n_groups=2, test_size=0.3, random_state=seed
            ),
            seed,
        )


def iter_regression_dsets() -> Iterator[Tuple[str, object, int]]:
    seeds = range(10)
    loaders: List[Tuple[str, Callable[..., object]]] = [
        ("lawschool", badr.datasets.fetch_lawschool),
        ("parkinsons", badr.datasets.fetch_parkinsons),
        ("communities_and_crime", badr.datasets.fetch_communities_and_crime),
        ("student_performance", badr.datasets.fetch_studentperformance),
    ]
    for name, loader in loaders:
        for seed in seeds:
            yield (
                f"{name}_seed{seed}",
                loader(random_state=seed),
                seed,
            )


def run_classification_experiment() -> pd.DataFrame:
    model_factory = lambda: badr.models.LogisticRegression(l2_reg=1e-2)
    metric_factory = lambda: badr.metrics.IndividualFairness()

    out_dir = os.path.join(os.path.dirname(__file__), "outputs")
    cls_records_path = os.path.join(out_dir, "classification_records.jsonl")
    completed = load_completed(cls_records_path, expected_methods=5)

    for dset_name, dset, seed in tqdm(
        iter_classification_dsets(),
        desc="Classification datasets",
        total=len(state_list) * 2,
    ):
        if dset_name in completed:
            print(f"Skipping {dset_name} (already completed)")
            continue
        try:
            dataset_records = evaluate_dataset(
                dset_name=dset_name,
                dset=dset,
                task="classification",
                model_factory=model_factory,
                metric_factory=metric_factory,
                badr_samples=200,
                seed=seed,
            )
        except MemoryError:
            print(f"MemoryError on {dset_name}; freeing and skipping.")
            del dset
            gc.collect()
            jax.clear_caches()
            continue

        append_records(cls_records_path, dataset_records)
        print_instance_table(
            dataset_records, task="classification", dset_name=dset_name
        )
        del dataset_records
        del dset
        gc.collect()
        jax.clear_caches()

    # If script crashed earlier, regenerate summary from on-disk records
    df = pd.read_json(cls_records_path, lines=True)
    return df


def run_regression_experiment() -> pd.DataFrame:
    model_factory = lambda: badr.models.RidgeRegression(l2_reg=1e-1)
    metric_factory = lambda: badr.metrics.DemographicParity()

    out_dir = os.path.join(os.path.dirname(__file__), "outputs")
    reg_records_path = os.path.join(out_dir, "regression_records.jsonl")
    completed = load_completed(reg_records_path, expected_methods=5)

    for dset_name, dset, seed in tqdm(
        iter_regression_dsets(),
        desc="Regression datasets",
        total=40,
    ):
        if dset_name in completed:
            print(f"Skipping {dset_name} (already completed)")
            continue
        try:
            dataset_records = evaluate_dataset(
                dset_name=dset_name,
                dset=dset,
                task="regression",
                model_factory=model_factory,
                metric_factory=metric_factory,
                badr_samples=200,
                seed=seed,
            )
        except MemoryError:
            print(f"MemoryError on {dset_name}; freeing and skipping.")
            del dset
            gc.collect()
            jax.clear_caches()
            continue

        append_records(reg_records_path, dataset_records)
        print_instance_table(dataset_records, task="regression", dset_name=dset_name)
        del dataset_records
        del dset
        gc.collect()
        jax.clear_caches()

    df = pd.read_json(reg_records_path, lines=True)
    return df


def main() -> None:
    print(
        "Running classification experiment (LogReg + Individual Fairness)...",
        flush=True,
    )
    cls_df = run_classification_experiment()
    cls_summary = summarize(cls_df, score_col="test_acc").round(4)
    print(cls_summary)

    print("\nRunning regression experiment (Ridge + Demographic Parity)...", flush=True)
    reg_df = run_regression_experiment()
    reg_summary = summarize(reg_df, score_col="test_rmse").round(4)
    print(reg_summary)

    out_dir = os.path.join(os.path.dirname(__file__), "outputs")
    os.makedirs(out_dir, exist_ok=True)
    cls_df.to_json(
        os.path.join(out_dir, "classification_records.jsonl"),
        orient="records",
        lines=True,
    )
    reg_df.to_json(
        os.path.join(out_dir, "regression_records.jsonl"), orient="records", lines=True
    )
    cls_summary.to_json(os.path.join(out_dir, "classification_summary.json"))
    reg_summary.to_json(os.path.join(out_dir, "regression_summary.json"))


if __name__ == "__main__":
    main()
