import sys
import os
import itertools

repo_root = os.path.abspath(os.path.join(os.getcwd(), "../.."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)
import badr
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

import logging
import json
from datetime import datetime
from collections import defaultdict
from pathlib import Path

RESULTS_JSONL_PATH = "results.jsonl"
COMPLETED_INDEX_PATH = "results_completed_keys.json"
LOG_JSON_PATH = "logs.jsonl"
ALGO_KEYS = ["erm", "balanced", "one_fit", "minmax", "badr"]

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)


class JsonLinesFileHandler(logging.Handler):
    """
    Simple handler that writes each log record as one JSON object per line.
    """

    def __init__(self, filename):
        super().__init__()
        # append mode so multiple runs accumulate
        self.file = open(filename, "a", encoding="utf-8")

    def emit(self, record):
        try:
            log_entry = {
                "time": datetime.fromtimestamp(record.created).isoformat(),
                "level": record.levelname,
                "logger": record.name,
                "message": record.getMessage(),
                # You can add more fields if you like:
                # "filename": record.filename,
                # "lineno": record.lineno,
                # "funcName": record.funcName,
            }
            self.file.write(json.dumps(log_entry) + "\n")
            self.file.flush()
        except Exception:
            self.handleError(record)

    def close(self):
        try:
            self.file.close()
        finally:
            super().close()


# Attach JSON handler to root logger
json_handler = JsonLinesFileHandler(LOG_JSON_PATH)
logging.getLogger().addHandler(json_handler)


def load_or_init_results(path=RESULTS_JSONL_PATH, index_path=COMPLETED_INDEX_PATH):
    """
    Load existing JSONL results (if any) and return:
        - completed_keys: set of (task, state, year, n_groups, metric) for which all algorithms are present.
        - per_key_algos: mapping key -> set of algorithms already recorded.
        - total_records: number of lines parsed (for logging only).
    """
    per_key_algos = defaultdict(set)
    total_records = 0

    # Fast path: load precomputed index of per_key_algos if present
    if os.path.exists(index_path):
        try:
            with open(index_path, "r", encoding="utf-8") as f:
                stored = json.load(f)
            for key_str, algos in stored.get("per_key_algos", {}).items():
                # keys stored as "task|state|year|n_groups|metric"
                parts = key_str.split("|")
                if len(parts) != 5:
                    continue
                task, state, year, n_groups, metric = parts
                key = (task, state, int(year), int(n_groups), metric)
                per_key_algos[key] = set(algos)
            total_records = stored.get("total_records", 0)
        except Exception:
            per_key_algos.clear()
            total_records = 0

    # Fallback: scan JSONL if index missing or empty
    if not per_key_algos:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    total_records += 1
                    algo = rec.get("algo")
                    key = (
                        rec.get("task"),
                        rec.get("state"),
                        rec.get("year"),
                        rec.get("n_groups"),
                        rec.get("metric"),
                    )
                    if algo:
                        per_key_algos[key].add(algo)

    completed_keys = {
        key for key, algos in per_key_algos.items() if set(ALGO_KEYS).issubset(algos)
    }

    return completed_keys, per_key_algos, total_records


state_list = [
    "WY",  # Wyoming
    "VT",  # Vermont
    "ND",  # North Dakota
    "SD",  # South Dakota
    "DE",  # Delaware
    "MT",  # Montana
    "ME",  # Maine
    "WV",  # West Virginia
    "ID",  # Idaho
    "PR",  # Puerto Rico
]

# state_list = [
#     "AL",
#     "AK",
#     "AR",
#     "CO",
#     "CT",
#     "DE",
#     "HI",
#     "ID",
#     "IN",
#     "IA",
#     "KS",
#     "KY",
#     "LA",
#     "ME",
#     "MD",
#     "MA",
#     "MN",
#     "MS",
#     "MO",
#     "MT",
#     "NE",
#     "NV",
#     "NH",
#     "NM",
#     "ND",
#     "OK",
#     "OR",
#     "RI",
#     "SC",
#     "SD",
#     "UT",
#     "VT",
#     "VA",
#     "WA",
#     "WV",
#     "WI",
#     "WY",
#     "PR",
# ]
years = [2014, 2015, 2016, 2017, 2018]
metrics = [  # for classification only
    badr.metrics.DemographicParity(),
    badr.metrics.EqualizedOdds(),
    badr.metrics.EqualOpportunity(),
    badr.metrics.DisparateMistreatment(),
    badr.metrics.GroupVariance(),
    badr.metrics.IndividualFairness(),
]
model = badr.models.LogisticRegression(l2_reg=1e-2)
for metric in metrics:
    metric.set_model(model)


def load_acs(task, state, year, n_groups):
    fetch_map = {
        "E": badr.datasets.fetch_ACSEmployment,
        "I": badr.datasets.fetch_ACSIncome,
        "TT": badr.datasets.fetch_ACSTravelTime,
    }
    if task not in fetch_map:
        raise ValueError("Unknown task")
    try:
        return fetch_map[task](states=state, year=year, n_groups=n_groups)
    except ValueError:
        return None


def process_single_dataset(
    dset,
    model,
    metrics,
    base_meta,
    completed_keys,
    per_key_algos,
    jsonl_handle,
    cache_batch_size=100,
):
    """
    Process one (task, state, year, n_groups_actual) dataset for all metrics & algorithms,
    reusing fits whenever possible.
    """
    # Use the actual number of groups from the dataset
    actual_n_groups = len(dset.X_train_list)
    meta_base = base_meta.copy()
    meta_base["n_groups"] = actual_n_groups

    # Caches so we don't refit the same model over and over
    erm_cache = None  # (weights, coef)
    balanced_cache = None  # (weights, coef)
    minmax_cache = None  # (weights, coef)
    onefit_cache = (
        None  # (eye_matrix, coefs_per_group)  coefs_per_group[i] = coef for w=e_i
    )

    write_buffer = []

    def flush_buffer():
        nonlocal write_buffer
        if not write_buffer:
            return
        jsonl_handle.write("".join(write_buffer))
        jsonl_handle.flush()
        write_buffer = []

    def _append_result(algo_name, metric_name, train_val, test_val, weights):
        rec = {
            "algo": algo_name,
            "metric": metric_name,
            "task": meta_base["task"],
            "state": meta_base["state"],
            "year": meta_base["year"],
            "n_groups": meta_base["n_groups"],
            "train_metric": float(train_val),
            "test_metric": float(test_val),
            "weights": np.asarray(weights, dtype=float).tolist(),
        }
        write_buffer.append(json.dumps(rec) + "\n")
        if len(write_buffer) >= cache_batch_size:
            flush_buffer()
        key = (
            rec["task"],
            rec["state"],
            rec["year"],
            rec["n_groups"],
            rec["metric"],
        )
        per_key_algos[key].add(algo_name)
        if set(ALGO_KEYS).issubset(per_key_algos[key]):
            completed_keys.add(key)

    # Predefine mm_specs here so we can reuse it inside
    mm_specs = {
        badr.models.LogisticRegression: (
            badr.models.NonsmoothMinMaxLogisticRegression,
            "list",
        ),
        badr.models.SVM: (badr.models.NSMMSVM, "flat"),
        badr.models.RidgeRegression: (badr.models.NSMMRR, "list"),
    }

    # Cache metric evaluations keyed by (metric_name, coef_bytes, split)
    metric_eval_cache = {}

    def cached_metric_eval(metric_obj, coef_arr, split):
        key = (metric_obj.name, coef_arr.tobytes(), split)
        val = metric_eval_cache.get(key)
        if val is not None:
            return val
        out = metric_obj.fun(coef_arr, dset, train_test=split).item()
        metric_eval_cache[key] = out
        return out

    def grid_search_weights(metric_obj, n_points=101):
        """Brute-force simplex grid search mirroring experiment_1 badr_vals."""
        n = len(dset.X_train_list)
        grid = jnp.linspace(0.0, 1.0, num=n_points)
        best_train = jnp.inf
        best_w = None
        best_coef = None

        for w_tuple in itertools.product(grid, repeat=n):
            w = jnp.array(w_tuple)
            if not jnp.isclose(w.sum(), 1.0):
                continue
            model.set_group_weights(w)
            model.fit(dset.X_train, dset.y_train, dset.groups)
            coef = jnp.array(model.coef_)
            train_val = float(metric_obj.fun(coef, dset, train_test="train").item())
            if train_val < best_train:
                best_train = train_val
                best_w = w
                best_coef = coef

        if best_w is None or best_coef is None:
            raise ValueError("Grid search failed to find feasible weights")

        test_val = float(metric_obj.fun(best_coef, dset, train_test="test").item())
        return best_w, best_coef, best_train, test_val

    for metric in metrics:
        metric_name = metric.name
        key = (
            meta_base["task"],
            meta_base["state"],
            meta_base["year"],
            meta_base["n_groups"],
            metric_name,
        )

        if key in completed_keys:
            logging.info(f"Skipping already computed combination: {key}")
            continue

        logging.info(
            f"Computing ERM, Balanced, One-Fit, MinMax, Badr for "
            f"task={meta_base['task']}, state={meta_base['state']}, "
            f"year={meta_base['year']}, n_groups={meta_base['n_groups']}, "
            f"metric={metric_name}."
        )

        # ---------- ERM (metric-independent weights, reuse across metrics) ----------
        if erm_cache is None:
            n = len(dset.X_train_list)
            w = jnp.ones(n) / n
            model.set_group_weights(w)
            model.fit(dset.X_train, dset.y_train, dset.groups)
            coef = jnp.array(model.coef_)
            erm_cache = (w, coef)
        else:
            w, coef = erm_cache

        erm_train = cached_metric_eval(metric, coef, "train")
        erm_test = cached_metric_eval(metric, coef, "test")
        _append_result("erm", metric_name, erm_train, erm_test, w)

        # ---------- Balanced (metric-independent weights, reuse across metrics) ----------
        if balanced_cache is None:
            sizes = jnp.array([X.shape[0] for X in dset.X_train_list])
            weights = 1.0 / sizes
            w_bal = weights / weights.sum()
            model.set_group_weights(w_bal)
            model.fit(dset.X_train, dset.y_train, dset.groups)
            coef_bal = jnp.array(model.coef_)
            balanced_cache = (w_bal, coef_bal)
        else:
            w_bal, coef_bal = balanced_cache

        bal_train = cached_metric_eval(metric, coef_bal, "train")
        bal_test = cached_metric_eval(metric, coef_bal, "test")
        _append_result("balanced", metric_name, bal_train, bal_test, w_bal)

        # ---------- One-Fit (pre-fit per group, re-use across metrics) ----------
        if onefit_cache is None:
            n = len(dset.X_train_list)
            eye = jnp.eye(n)
            coefs = []
            for i in range(n):
                w_i = eye[i]
                model.set_group_weights(w_i)
                model.fit(dset.X_train, dset.y_train, dset.groups)
                coefs.append(jnp.array(model.coef_))
            coefs = jnp.stack(coefs)  # shape: (n_groups, n_params)
            onefit_cache = (eye, coefs)

        eye, coefs = onefit_cache

        # For this metric, pick the group-specific model with smallest train_metric
        # Vectorized scoring if metric supports batch; otherwise fallback to loop
        try:
            train_vals = metric.fun(coefs, dset, train_test="train").ravel()
            train_vals = np.asarray(train_vals, dtype=float)
        except Exception:
            train_vals = np.array(
                [
                    cached_metric_eval(metric, coefs[i], "train")
                    for i in range(coefs.shape[0])
                ],
                dtype=float,
            )

        best_idx = int(np.argmin(train_vals))
        best_w = eye[best_idx]
        best_coef = coefs[best_idx]
        best_train = float(train_vals[best_idx])
        best_test = cached_metric_eval(metric, best_coef, "test")
        _append_result("one_fit", metric_name, best_train, best_test, best_w)

        # ---------- MinMax (metric-independent weights, reuse across metrics) ----------
        if minmax_cache is None:
            for base_cls, (mm_cls, fit_kind) in mm_specs.items():
                if isinstance(model, base_cls):
                    mm_model = mm_cls(l2_reg=model.l2_reg)
                    if fit_kind == "list":
                        mm_model.fit(dset.X_train_list, dset.y_train_list)
                    else:
                        mm_model.fit(dset.X_train, dset.y_train, dset.groups)
                    break
            else:
                raise ValueError("Unsupported model type for minmax_vals")

            gw = jnp.asarray(mm_model.group_weights_, dtype=jnp.float64)
            model.set_group_weights(gw)
            model.fit(dset.X_train, dset.y_train, dset.groups)
            coef_mm = jnp.array(model.coef_)
            minmax_cache = (gw, coef_mm)

        gw, coef_mm = minmax_cache
        mm_train = cached_metric_eval(metric, coef_mm, "train")
        mm_test = cached_metric_eval(metric, coef_mm, "test")
        _append_result("minmax", metric_name, mm_train, mm_test, gw)

        # ---------- Badr via grid search (mirrors experiment_1 badr_vals) ----------
        gw_badr, coef_badr, ar_train, ar_test = grid_search_weights(metric)
        model.set_group_weights(gw_badr)
        _append_result("badr", metric_name, ar_train, ar_test, gw_badr)

        logging.info("Saved intermediate results to results.jsonl")

    flush_buffer()


def main():
    # Load previous results (if any) and figure out what is already done
    completed_keys, per_key_algos, total_records = load_or_init_results(
        RESULTS_JSONL_PATH
    )
    logging.info(f"Loaded {total_records} existing results (JSONL lines).")
    logging.info(
        f"{len(completed_keys)} (task,state,year,n_groups,metric) combinations already complete."
    )

    tasks = ["E", "I", "TT"]

    # Open once in append mode; each _append_result flushes
    with open(RESULTS_JSONL_PATH, "a", encoding="utf-8") as jsonl_handle:
        for task in tasks:
            for year in years:
                for state in state_list:
                    logging.info(
                        f"=== Starting task={task}, state={state}, year={year} ==="
                    )
                    n_groups = 2
                    logging.info(
                        f"  Using fixed n_groups={n_groups} for task={task}, state={state}, year={year}."
                    )
                    dset = load_acs(task, state, year, n_groups=n_groups)

                    if dset is None:
                        logging.info(
                            f"  Dataset unavailable for task={task}, state={state}, year={year} "
                            f"with n_groups={n_groups}; skipping."
                        )
                        continue

                    base_meta = {
                        "task": task,
                        "state": state,
                        "year": year,
                    }

                    process_single_dataset(
                        dset=dset,
                        model=model,
                        metrics=metrics,
                        base_meta=base_meta,
                        completed_keys=completed_keys,
                        per_key_algos=per_key_algos,
                        jsonl_handle=jsonl_handle,
                    )

                    # let this dataset be garbage-collected before moving on
                    del dset

    # Persist sidecar index for fast resume
    try:
        per_key_algos_serializable = {
            f"{k[0]}|{k[1]}|{k[2]}|{k[3]}|{k[4]}": sorted(list(v))
            for k, v in per_key_algos.items()
        }
        with open(COMPLETED_INDEX_PATH, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "per_key_algos": per_key_algos_serializable,
                    "total_records": total_records,
                },
                f,
            )
    except Exception:
        logging.warning(
            "Failed to persist completed index; will fall back to scanning JSONL next run."
        )


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logging.exception("Script crashed with an exception.")
        raise
