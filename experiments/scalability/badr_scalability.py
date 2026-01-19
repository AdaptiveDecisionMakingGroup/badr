from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import json
import jax
import jax.numpy as jnp
from jax import ShapeDtypeStruct
import numpy as np
from tqdm import tqdm

import sys
import os

repo_root = os.path.abspath(os.path.join(os.getcwd(), "../.."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)
import badr

CONFIG: Dict[str, Any] = {
    "n_runs": 5,
    "seed_start": 100,
    "sizes": {
        "slsqp": [int(x) for x in np.linspace(5000, 23000, num=10)],
        "badr": [
            int(x)
            for x in np.hstack(
                (np.linspace(5000, 23000, num=10), np.linspace(25000, 150000, num=26))
            )
        ],
    },
    "n_iter": 150,
    "badr": {
        "batch_divisor": 20,
        "step_w": 0.1,
        "step_v": 0.1,
        "step_lambda": 2.0,
        "clip_value": 1.0,
    },
    "grid_points": 120,
    "output_path": "fat_experiment.json",
    "model": {"l2_reg": 1e-1},
    "dataset": {"states": "CA", "year": 2018, "n_groups": 2, "test_size": 0.05},
}

CONFIG["random_seeds"] = list(
    range(CONFIG["seed_start"], CONFIG["seed_start"] + CONFIG["n_runs"])
)

RESULTS: Dict[int, Dict[str, Any]] = {}


def human_size(size: int) -> str:
    if size == 0:
        return "0"
    if size % 1000 == 0:
        return f"{size // 1000}k"
    return f"{size:,}"


def make_run_key(run_idx: int, seed: int) -> str:
    return f"run_{run_idx + 1}_seed_{seed}"


def get_min(oracle: Any) -> float:
    xs = jnp.linspace(0.0, 1.0, CONFIG["grid_points"], dtype=jnp.float32)
    lam_grid = jnp.stack([xs, 1.0 - xs], axis=1)

    def call_oracle(lam: jnp.ndarray) -> jnp.ndarray:
        def _eval(lam_host: np.ndarray) -> np.ndarray:
            lam_np = np.asarray(lam_host, dtype=np.float64)
            value = oracle.fun(lam_np)
            return np.asarray(value, dtype=np.float32).reshape(())

        return jax.pure_callback(_eval, ShapeDtypeStruct((), jnp.float32), lam)

    values = jax.vmap(call_oracle)(lam_grid)
    return float(jnp.min(values))


def evaluate_with_oracle(oracle: Any, lam: np.ndarray) -> float:
    value = oracle.fun(jnp.asarray(lam, dtype=jnp.float64))
    arr = np.asarray(value, dtype=float).reshape(-1)
    return float(arr[0]) if arr.size else float("nan")


def collect_trace(solver: Any, eval_oracle: Any) -> Dict[str, Any]:
    lambdas_raw: List[np.ndarray] = []
    history_lambda = getattr(solver, "history_lambda", []) or []
    for lam in history_lambda:
        lambdas_raw.append(np.asarray(lam, dtype=float).reshape(-1))
    if not lambdas_raw:
        final_lambda = getattr(solver, "lambda_", None)
        if final_lambda is not None:
            lambdas_raw.append(np.asarray(final_lambda, dtype=float).reshape(-1))
    step_seconds = [float(t) for t in (getattr(solver, "history_time", []) or [])]
    cumulative = np.cumsum(step_seconds).tolist() if step_seconds else []
    implicit_values = [evaluate_with_oracle(eval_oracle, lam) for lam in lambdas_raw]
    return {
        "lambda": [lam.tolist() for lam in lambdas_raw],
        "step_seconds": step_seconds,
        "cumulative_seconds": cumulative,
        "implicit_value": implicit_values,
    }


def monotonize(arr: List[float]) -> List[float]:
    mono_arr: List[float] = []
    current_min = float("inf")
    for value in arr:
        if value < current_min:
            current_min = value
        mono_arr.append(current_min)
    return mono_arr


def build_oracles(dataset: Any) -> tuple[Any, Any]:
    model = badr.models.LogisticRegression(l2_reg=CONFIG["model"]["l2_reg"])
    metric = badr.metrics.IndividualFairness()
    implicit = badr.oracles.ImplicitOracle(dataset, model, metric)
    stochastic = badr.oracles.StochasticOracle(dataset, model, metric)
    return implicit, stochastic


def save_results(output_path: Path) -> None:
    with output_path.open("w") as f:
        json.dump(RESULTS, f, indent=2)


def run_slsqp_experiments(
    base_dataset: Any, output_path: Path, sizes: List[int] | None = None
) -> None:
    plan = [int(x) for x in (sizes if sizes is not None else CONFIG["sizes"]["slsqp"])]
    for size in tqdm(plan, desc="Running SLSQP experiments", ncols=90):
        size_key = int(size)
        size_results = RESULTS.setdefault(size_key, {})
        for run_idx, seed in enumerate(CONFIG["random_seeds"]):
            dataset = base_dataset.subsample(size, random_state=seed)
            implicit_train, _ = build_oracles(dataset)
            ymin = get_min(implicit_train)
            slsqp = badr.algorithms.SLSQP()
            slsqp.set_oracle(implicit_train)
            slsqp.run(max_iter=int(CONFIG["n_iter"] // 2), trace=True)
            slsqp_trace = collect_trace(slsqp, implicit_train)
            run_key = make_run_key(run_idx, seed)
            run_entry = size_results.setdefault(run_key, {})
            run_entry["slsqp"] = slsqp_trace
            run_entry["ymin"] = ymin
        save_results(output_path)


def run_badr_experiments(
    base_dataset: Any, output_path: Path, sizes: List[int] | None = None
) -> None:
    batch_divisor = CONFIG["badr"]["batch_divisor"]
    plan = [int(x) for x in (sizes if sizes is not None else CONFIG["sizes"]["badr"])]
    for size in tqdm(plan, desc="Running BADR experiments", ncols=90):
        size_key = int(size)
        size_results = RESULTS.setdefault(size_key, {})
        for run_idx, seed in enumerate(CONFIG["random_seeds"]):
            print(
                "Currently running the "
                f"{human_size(size_key)} sample size experiments "
                f"({run_idx + 1}/{len(CONFIG['random_seeds'])})"
            )
            dataset = base_dataset.subsample(size, random_state=seed)
            implicit_train, stochastic = build_oracles(dataset)
            w0 = implicit_train.solve_lower(jnp.array([0.5, 0.5], dtype=jnp.float32))
            badr = badr.algorithms.BADRSGD(
                w0,
                batch_size=max(1, int(size // batch_divisor)),
                step_w=CONFIG["badr"]["step_w"],
                step_v=CONFIG["badr"]["step_v"],
                step_lambda=CONFIG["badr"]["step_lambda"],
                clip_value=CONFIG["badr"]["clip_value"],
            )
            badr.set_oracle(stochastic)
            badr.run(max_iter=int(CONFIG["n_iter"]), trace=True)
            badr_trace = collect_trace(badr, implicit_train)
            run_key = make_run_key(run_idx, seed)
            run_entry = size_results.setdefault(run_key, {})
            run_entry["badr"] = badr_trace
        save_results(output_path)


def pending_badr_sizes(min_size: int = 0) -> List[int]:
    """Return sample sizes that still need BADR traces for the configured seeds."""
    pending: List[int] = []
    for size in CONFIG["sizes"]["badr"]:
        size_key = int(size)
        if size_key < min_size:
            continue
        size_results = RESULTS.get(size_key, {})
        for run_idx, seed in enumerate(CONFIG["random_seeds"]):
            run_key = make_run_key(run_idx, seed)
            if "badr" not in size_results.get(run_key, {}):
                pending.append(size_key)
                break
    return pending


def max_completed_badr_size() -> int:
    completed = 0
    for size in sorted(int(x) for x in CONFIG["sizes"]["badr"]):
        size_results = RESULTS.get(size, {})
        if not size_results:
            break
        done = True
        for run_idx, seed in enumerate(CONFIG["random_seeds"]):
            run_key = make_run_key(run_idx, seed)
            if "badr" not in size_results.get(run_key, {}):
                done = False
                break
        if not done:
            break
        completed = size
    return completed


def main() -> None:
    output_path = Path(CONFIG["output_path"])
    base_dataset = badr.datasets.fetch_ACSEmployment(**CONFIG["dataset"])
    if output_path.exists():
        with output_path.open("r") as f:
            persisted = json.load(f)
        for size_key, runs in persisted.items():
            RESULTS[int(size_key)] = runs
        max_size = max_completed_badr_size()
        target_size = max(int(x) for x in CONFIG["sizes"]["badr"])
        print(
            "Current state of experiments: SLSQP done. BADR reached "
            f"{human_size(max_size)}/{human_size(target_size)}"
        )
    if output_path.exists():
        max_size = max_completed_badr_size()
        todo = pending_badr_sizes(min_size=max_size + 1)
        if todo:
            run_badr_experiments(base_dataset, output_path, sizes=todo)
    else:
        run_slsqp_experiments(base_dataset, output_path)
        run_badr_experiments(base_dataset, output_path)


if __name__ == "__main__":
    main()
