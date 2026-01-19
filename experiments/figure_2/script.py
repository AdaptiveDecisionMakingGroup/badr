from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from scipy.optimize import minimize
from tqdm.auto import tqdm
from tueplots import axes as tp_axes
from tueplots import bundles, figsizes, fontsizes

# Ensure the repository root is importable when running the script directly
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import badr


METHOD_COLORS = {
    "ERM": "#662C91",
    "Balanced": "#DA9F93",
    "One-Fit": "#A0AF63",
    "MinMax": "#6699CC",
    "Badr": "#C81919",
}


def _to_serializable(obj: Any):
    """Recursively convert JAX/NumPy containers to JSON-serializable structures."""

    if isinstance(obj, (np.ndarray, jnp.ndarray)):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_serializable(v) for v in obj]
    return obj


def _save_json(obj: Any, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(_to_serializable(obj), f, indent=2)


def _load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


class PenalizedLR:
    def __init__(
        self,
        penalty: float,
        l2_reg: float,
        fit_intercept: bool = True,
        tol: float = 1e-5,
        max_iter: int = 200,
    ):
        self.penalty = penalty
        self.l2_reg = l2_reg
        self.fit_intercept = fit_intercept
        self.tol = tol
        self.max_iter = max_iter
        self.base_model = badr.models.LogisticRegression(
            l2_reg=self.l2_reg,
            fit_intercept=self.fit_intercept,
        )
        self.coef_: jnp.ndarray | None = None
        self.lmbd = 0.5

    def set_lmbd(self, lmbd: float):
        self.lmbd = lmbd
        return self

    def set_metric(self, metric):
        self.metric = metric
        return self

    def _group_loss(
        self, w: jnp.ndarray, dset: "badr.datasets.Dataset", train_test: str
    ) -> tuple:
        arr = self.base_model._group_loss(dset, train_test, w)
        return arr[0], arr[1]

    def _objective(
        self, w: jnp.ndarray, dset: "badr.datasets.Dataset", train_test: str
    ) -> jnp.ndarray:
        loss_g1, loss_g2 = self._group_loss(w, dset, train_test)
        if self.fit_intercept:
            coef = w[1:]
        else:
            coef = w
        fair = self.metric.fun(coef, dset, train_test)
        ridge = 0.5 * self.l2_reg * jnp.dot(coef, coef)
        combined = (
            self.lmbd * loss_g1
            + (1.0 - self.lmbd) * loss_g2
            + self.penalty * fair
            + ridge
        )
        return combined

    def minimize_loss(
        self,
        dset: "badr.datasets.Dataset",
        w0: jnp.ndarray,
        train_test: str = "train",
        method: str = "BFGS",
    ):
        options = {
            "maxiter": self.max_iter,
        }

        def _loss(w):
            return self._objective(w, dset, train_test)

        res = minimize(fun=_loss, x0=w0, method=method, options=options)
        self.coef_ = res.x
        return self


# Pareto front over λ (pen=0)
def get_badr(dset, metric, model, n_points=80):
    metric.set_model(model)
    xs = np.linspace(0.0, 1.0, n_points + 1)
    ys = []
    l1_path, l2_path = [], []
    best = np.inf
    for lmbd in tqdm(xs, desc="Pareto front"):
        model.set_group_weights(jnp.array([lmbd, 1.0 - lmbd]))
        model.fit(dset.X_train, dset.y_train, dset.groups)
        fair = metric.fun(model.coef_, dset)
        ys.append(float(fair))
        arr = model._group_loss(dset, "train")
        l1_path.append(float(arr[0]))
        l2_path.append(float(arr[1]))
        if fair < best:
            best = fair
            arr = model._group_loss(dset, "train")
            x = float(arr[0])
            y = float(arr[1])
    return x, y, xs, np.array(ys), np.array(l1_path), np.array(l2_path)


def _eval_with_weights(model, metric, dset, weights):
    metric.set_model(model)
    model.set_group_weights(weights)
    model.fit(dset.X_train, dset.y_train, dset.groups)

    train_metric = float(metric.fun(model.coef_, dset, train_test="train"))
    test_metric = float(metric.fun(model.coef_, dset, train_test="test"))
    losses = model._group_loss(dset, "train")

    return (
        float(losses[0]),
        float(losses[1]),
        train_metric,
        test_metric,
        np.array(weights, dtype=float),
    )


def get_erm(dset, metric, model):
    n = len(dset.X_train_list)

    weights = jnp.ones(n) / n
    # ERM = uniform group weights

    return _eval_with_weights(model, metric, dset, weights)


def get_balanced(dset, metric, model):
    sizes = jnp.array([X.shape[0] for X in dset.X_train_list], dtype=float)

    weights = 1.0 / sizes
    weights = weights / weights.sum()

    return _eval_with_weights(model, metric, dset, weights)


def get_onefit(dset, metric, model):
    n = len(dset.X_train_list)

    candidates = []

    for i in range(n):
        w = jnp.eye(n)[i]

        res = _eval_with_weights(model, metric, dset, w)

        candidates.append(res)

    # pick weights that minimize train fairness metric

    best = min(candidates, key=lambda t: t[2])

    return best


def get_minmax(dset, metric, model):
    # Solve nonsmooth minmax to get group weights, then evaluate like others

    mm_model = badr.models.NonsmoothMinMaxLogisticRegression(l2_reg=model.l2_reg)

    mm_model.fit(dset.X_train_list, dset.y_train_list)

    gw = jnp.asarray(mm_model.group_weights_, dtype=jnp.float64)

    return _eval_with_weights(model, metric, dset, gw)


def generate_regularization_path(dset, metric, model, pen_grid):
    l1_path, l2_path = [], []
    prop = []
    for pen in tqdm(pen_grid, desc="Regularization path"):
        penalized_model = PenalizedLR(
            penalty=pen,
            l2_reg=model.l2_reg,
            fit_intercept=True,
            max_iter=300,
        )
        penalized_model.set_metric(metric)
        n_params = (
            dset.X_train.shape[1] + 1
            if penalized_model.fit_intercept
            else dset.X_train.shape[1]
        )
        penalized_model.set_lmbd(0.5)
        penalized_model.minimize_loss(
            dset,
            w0=jnp.zeros(n_params, dtype=jnp.float64),
            train_test="train",
        )
        w_sol = penalized_model.coef_
        w_metric = w_sol[1:] if penalized_model.fit_intercept else w_sol
        prop.append(float(metric.fun(w_metric, dset, "train")))
        arr = model._group_loss(dset, "train", w_sol)
        l1_path.append(float(arr[0]))
        l2_path.append(float(arr[1]))
    return np.array(l1_path), np.array(l2_path), np.array(prop)


def _densify_curve(x, y, z, factor):
    """
    Densify a parametric curve (x(t), y(t), z(t)) by linear interpolation.

    Parameters
    ----------
    x, y, z : 1D arrays of the same length
    factor : int >= 1
        Number of extra samples per original segment. If 1, nothing is changed.

    Returns
    -------
    x_d, y_d, z_d : 1D arrays
    """
    x = np.asarray(x)
    y = np.asarray(y)
    z = np.asarray(z)

    if factor <= 1 or x.size < 2:
        return x, y, z

    n = x.size
    t = np.linspace(0.0, 1.0, n)
    t_dense = np.linspace(0.0, 1.0, (n - 1) * factor + 1)

    x_d = np.interp(t_dense, t, x)
    y_d = np.interp(t_dense, t, y)
    z_d = np.interp(t_dense, t, z)
    return x_d, y_d, z_d


def build_data(dset, model, metric, lmbd_grid=None, pen_grid=None):
    """
    Build the two curves + method markers:
      - Pareto front at penalization=0 from `get_badr` (lambda sweep)
      - Penalization path over `pen_grid` from `generate_regularization_path`
      - Method markers (ERM, Balanced, One-Fit, MinMax, Badr) using the
        same definitions as experiment_1.

    Returns dict with grids, curves, Pareto paths, and `markers` mapping
    method name -> (loss_g1, loss_g2).
    """
    metric.set_model(model)
    if lmbd_grid is None:
        lmbd_grid = np.linspace(0.0, 1.0, 80)
    if pen_grid is None:
        pen_grid = np.logspace(-6, -0.7, 10)

    xrp, yrp, frp = generate_regularization_path(dset, metric, model, pen_grid)

    # badr Pareto front at penalty=0
    xbadr, ybadr, xs_badr, ys_badr, l1_badr, l2_badr = get_badr(
        dset, metric, model, n_points=101
    )
    ys_badr = (ys_badr - ys_badr.min()) / (ys_badr.max() - ys_badr.min() + 1e-12)

    erm_l1, erm_l2, erm_train, erm_test, erm_w = get_erm(dset, metric, model)
    bal_l1, bal_l2, bal_train, bal_test, bal_w = get_balanced(dset, metric, model)
    one_l1, one_l2, one_train, one_test, one_w = get_onefit(dset, metric, model)
    mm_l1, mm_l2, mm_train, mm_test, mm_w = get_minmax(dset, metric, model)

    markers = {
        "ERM": (erm_l1, erm_l2),
        "Balanced": (bal_l1, bal_l2),
        "One-Fit": (one_l1, one_l2),
        "MinMax": (mm_l1, mm_l2),
        "Badr": (xbadr, ybadr),
    }

    marker_details = {
        "ERM": {
            "loss1": erm_l1,
            "loss2": erm_l2,
            "train_metric": erm_train,
            "test_metric": erm_test,
            "weights": erm_w,
        },
        "Balanced": {
            "loss1": bal_l1,
            "loss2": bal_l2,
            "train_metric": bal_train,
            "test_metric": bal_test,
            "weights": bal_w,
        },
        "One-Fit": {
            "loss1": one_l1,
            "loss2": one_l2,
            "train_metric": one_train,
            "test_metric": one_test,
            "weights": one_w,
        },
        "MinMax": {
            "loss1": mm_l1,
            "loss2": mm_l2,
            "train_metric": mm_train,
            "test_metric": mm_test,
            "weights": mm_w,
        },
        "Badr": {
            "loss1": xbadr,
            "loss2": ybadr,
            "train_metric": None,
            "test_metric": None,
            "weights": None,
        },
    }

    return {
        "pen_grid": pen_grid,
        "lmbd_grid": lmbd_grid,
        "xrp": xrp,
        "yrp": yrp,
        "frp": frp,
        "xbadr": xbadr,
        "ybadr": ybadr,
        "xs_badr": xs_badr,
        "ys_badr": ys_badr,
        "badr_l1_path": l1_badr,
        "badr_l2_path": l2_badr,
        "markers": markers,
        "marker_details": marker_details,
    }
    # res = build_data(dset, model, metric)


def fairness_values(model, metric, dset, weights):
    metric.set_model(model)
    model.set_group_weights(weights)
    model.fit(dset.X_train, dset.y_train, dset.groups)
    train_metric = float(metric.fun(model.coef_, dset, train_test="train").item())
    test_metric = float(metric.fun(model.coef_, dset, train_test="test").item())
    return {
        "train_metric": train_metric,
        "test_metric": test_metric,
    }


def erm_vals(model, metric, dset):
    weights = jnp.ones(len(dset.X_train_list)) / len(dset.X_train_list)
    dct = fairness_values(model, metric, dset, weights)
    dct["weights"] = np.array(weights, dtype=float)
    return dct


def one_fit_vals(model, metric, dset):
    n = len(dset.X_train_list)
    vals = [
        (
            fairness_values(model, metric, dset, jnp.eye(n)[i])["train_metric"],
            jnp.eye(n)[i],
        )
        for i in range(n)
    ]
    _, best_w = min(vals, key=lambda x: x[0])
    dct = fairness_values(model, metric, dset, best_w)
    dct["weights"] = np.array(best_w, dtype=float)
    return dct


def balanced_vals(model, metric, dset):
    sizes = jnp.array([X.shape[0] for X in dset.X_train_list])
    weights = 1 / sizes
    weights = weights / weights.sum()
    dct = fairness_values(model, metric, dset, weights)
    dct["weights"] = np.array(weights, dtype=float)
    return dct


def badr_vals(model, metric, dset, n_points=101):
    n = len(dset.X_train_list)
    grid = jnp.linspace(0.0, 1.0, num=n_points)
    best_val = jnp.inf
    best_w = None
    import itertools

    for w_tuple in itertools.product(grid, repeat=n):
        w = jnp.array(w_tuple)
        if jnp.isclose(w.sum(), 1.0):
            val = fairness_values(model, metric, dset, w)["train_metric"]
            if val < best_val:
                best_val = val
                best_w = w
    dct = fairness_values(model, metric, dset, best_w)
    dct["weights"] = np.array(best_w, dtype=float)
    return dct


def minmax_vals(model, metric, dset):
    if isinstance(model, badr.models.LogisticRegression):
        mm_model = badr.models.NonsmoothMinMaxLogisticRegression(l2_reg=model.l2_reg)
        mm_model.fit(dset.X_train_list, dset.y_train_list)
    elif isinstance(model, badr.models.SVM):
        mm_model = badr.models.NSMMSVM(l2_reg=model.l2_reg)
        mm_model.fit(dset.X_train, dset.y_train, dset.groups)
    elif isinstance(model, badr.models.RidgeRegression):
        mm_model = badr.models.NSMMRR(l2_reg=model.l2_reg)
        mm_model.fit(dset.X_train_list, dset.y_train_list)
    else:
        raise ValueError("Unsupported model type for minmax_vals")
    gw = jnp.asarray(mm_model.group_weights_, dtype=jnp.float64)
    dct = fairness_values(model, metric, dset, gw)
    dct["weights"] = np.array(gw, dtype=float)
    return dct


def lineplot(model, metric, dset):
    xs = jnp.linspace(0.0, 1.0, num=101)
    ys = []
    for x in xs:
        weights = jnp.array([x, 1 - x])
        dct = fairness_values(model, metric, dset, weights)
        ys.append(dct["train_metric"])
    ys = jnp.array(ys)
    return xs, ys


def load_datasets():
    """Load the datasets used in figure 2."""

    return {
        "SD 2014": badr.datasets.fetch_ACSEmployment(states="SD", year=2014),
        "ND 2018": badr.datasets.fetch_ACSEmployment(states="ND", year=2018),
        "VT 2015": badr.datasets.fetch_ACSTravelTime(states="VT", year=2015),
    }


def init_models(dataset_names, l2_reg=1e-2):
    """Create one logistic regression model per dataset."""

    return {
        name: badr.models.LogisticRegression(l2_reg=l2_reg) for name in dataset_names
    }


def init_metrics(models):
    """Attach an IndividualFairness metric to each model."""

    metrics = {}
    for name, model in models.items():
        metric = badr.metrics.IndividualFairness()
        metric.set_model(model)
        metrics[name] = metric
    return metrics


def build_data_for_datasets(dsets, models, metrics):
    return {
        name: build_data(dset, models[name], metrics[name])
        for name, dset in dsets.items()
    }


def build_results(dsets, models, metrics):
    results = {}
    for dset_name, dset in dsets.items():
        model = models[dset_name]
        metric = metrics[dset_name]
        results[dset_name] = {
            "ERM": erm_vals(model, metric, dset),
            "One-Fit": one_fit_vals(model, metric, dset),
            "Balanced": balanced_vals(model, metric, dset),
            "Badr": badr_vals(model, metric, dset, n_points=101),
            "MinMax": minmax_vals(model, metric, dset),
            "LinePlot": lineplot(model, metric, dset),
        }
    return results


def _plot_row_one_on_axes(axes, data_dict, alpha=0.9, cmap="RdBu_r", norm=None):
    """
    Draw the top row on existing axes.
    Returns: (cmap, norm, legend_handles, legend_labels)
    """
    keys = list(data_dict.keys())

    method_colors = {
        key: METHOD_COLORS[key]
        for key in ["Badr", "ERM", "Balanced", "One-Fit", "MinMax"]
    }
    marker_styles = {
        "Badr": dict(
            marker="X",
            s=40,
            zorder=5,
            color=method_colors["Badr"],
            alpha=alpha,
            lw=0.5,
            edgecolor="black",
        ),
        "ERM": dict(
            marker="v",
            s=30,
            zorder=3,
            color=method_colors["ERM"],
            alpha=alpha,
            lw=0.25,
            edgecolor="black",
        ),
        "Balanced": dict(
            marker="^",
            s=30,
            zorder=3,
            color=method_colors["Balanced"],
            alpha=alpha,
            lw=0.25,
            edgecolor="black",
        ),
        "One-Fit": dict(
            marker="o",
            s=30,
            zorder=4,
            color=method_colors["One-Fit"],
            alpha=alpha,
            lw=0.25,
            edgecolor="black",
        ),
        "MinMax": dict(
            marker="p",
            s=30,
            zorder=3,
            color=method_colors["MinMax"],
            alpha=alpha,
            lw=0.25,
            edgecolor="black",
        ),
    }
    method_labels = {
        "Badr": "badr",
        "ERM": "Uniform sampling",
        "Balanced": "Balanced sampling",
        "One-Fit": "One-group fitting",
        "MinMax": "Minimax fairness",
    }

    if norm is None:
        all_ys = [np.asarray(d["ys_badr"]) for d in data_dict.values()]
        all_frp = [np.asarray(d["frp"]) for d in data_dict.values()]
        all_cvals = np.concatenate(all_ys + all_frp)
        norm = plt.Normalize(vmin=all_cvals.min(), vmax=all_cvals.max())

    legend_handles = []
    legend_labels = []
    seen_labels = set()

    for i, key in enumerate(keys):
        d = data_dict[key]
        ax = axes[i]

        required_keys = {
            "badr_l1_path",
            "badr_l2_path",
            "xrp",
            "yrp",
            "xbadr",
            "ybadr",
            "ys_badr",
            "markers",
            "frp",
        }
        missing = required_keys - d.keys()
        if missing:
            raise ValueError(
                f"data_dict['{key}'] missing keys {missing}. "
                "Rebuild data_dict with the new fields."
            )

        l1 = np.asarray(d["badr_l1_path"])
        l2 = np.asarray(d["badr_l2_path"])
        cvals_pf = np.asarray(d["ys_badr"])

        # Densify Pareto front for smoother display
        n_pf = l1.shape[0]
        t_pf = np.linspace(0.0, 1.0, n_pf)
        t_pf_dense = np.linspace(0.0, 1.0, max(200, n_pf * 5))
        l1_dense = np.interp(t_pf_dense, t_pf, l1)
        l2_dense = np.interp(t_pf_dense, t_pf, l2)
        cvals_pf_dense = np.interp(t_pf_dense, t_pf, cvals_pf)

        points_pf = np.column_stack([l2_dense, l1_dense])
        segments_pf = np.stack([points_pf[:-1], points_pf[1:]], axis=1)

        lc_pf = LineCollection(
            segments_pf,
            array=cvals_pf_dense[:-1],
            cmap=cmap,
            norm=norm,
            linewidth=2.0,
            zorder=2,
        )
        ax.add_collection(lc_pf)

        xrp = np.asarray(d["xrp"])
        yrp = np.asarray(d["yrp"])
        frp = np.asarray(d["frp"])

        n_pts = xrp.shape[0]
        t = np.linspace(0.0, 1.0, n_pts)
        t_dense = np.linspace(0.0, 1.0, 200)

        x_dense = np.interp(t_dense, t, xrp)
        y_dense = np.interp(t_dense, t, yrp)
        frp_dense = np.interp(t_dense, t, frp)

        points_rp = np.column_stack([y_dense, x_dense])
        segments_rp = np.stack([points_rp[:-1], points_rp[1:]], axis=1)

        lc_rp = LineCollection(
            segments_rp,
            array=frp_dense[:-1],
            cmap=cmap,
            norm=norm,
            linewidth=2.0,
            zorder=1,
        )
        ax.add_collection(lc_rp)

        for method in ["Badr", "ERM", "Balanced", "One-Fit", "MinMax"]:
            if method not in d["markers"]:
                continue
            mx, my = d["markers"][method]
            style = marker_styles[method]
            lbl = method_labels[method]
            sc = ax.scatter(my, mx, label=lbl, **style)
            if lbl not in seen_labels:
                legend_handles.append(sc)
                legend_labels.append(lbl)
                seen_labels.add(lbl)

        ax.set_title(key, pad=2)
        ax.set_xlabel("Loss group 2")
        if i == 0:
            ax.set_ylabel("Loss group 1")

    return cmap, norm, legend_handles, legend_labels


def _plot_row_two_on_axes(axes, results, dsets, alpha=0.9):
    """
    Bottom row: one axis per dataset.

    axes  : 1D iterable of Axes (len == number of datasets)
    results : dict keyed by (dset_name, metric_name) or dset_name
    dsets : dict mapping dset_name -> dataset  OR iterable of (dset_name, dset)
    """
    axes_arr = np.asarray(axes).ravel()

    # normalise dsets to list of (name, dataset)
    if isinstance(dsets, dict):
        dataset_items = list(dsets.items())
    else:
        dataset_items = list(dsets)

    n_d = len(dataset_items)
    if axes_arr.size != n_d:
        raise ValueError(
            f"_plot_row_two_on_axes expects {n_d} axes, got {axes_arr.size}"
        )

    methods = [
        ("Badr", "badr"),
        ("Balanced", "Balanced sampling"),
        ("ERM", "Uniform sampling"),
        ("MinMax", "Minimax fairness"),
        ("One-Fit", "One-group fitting"),
    ]

    method_colors = METHOD_COLORS

    marker_styles = {
        "ERM": dict(
            marker="v",
            s=30,
            zorder=3,
            color=method_colors["ERM"],
            alpha=alpha,
            lw=0.25,
            edgecolor="black",
        ),
        "Balanced": dict(
            marker="^",
            s=30,
            zorder=3,
            color=method_colors["Balanced"],
            alpha=alpha,
            lw=0.25,
            edgecolor="black",
        ),
        "One-Fit": dict(
            marker="o",
            s=30,
            zorder=4,
            color=method_colors["One-Fit"],
            alpha=alpha,
            lw=0.25,
            edgecolor="black",
        ),
        "MinMax": dict(
            marker="p",
            s=30,
            zorder=3,
            color=method_colors["MinMax"],
            alpha=alpha,
            lw=0.25,
            edgecolor="black",
        ),
        "Badr": dict(
            marker="X",
            s=30,
            zorder=5,
            color=method_colors["Badr"],
            alpha=alpha,
            lw=0.25,
            edgecolor="black",
        ),
    }

    legend_handles = []
    legend_labels = []

    for ax, (dset_name, _dset) in zip(axes_arr, dataset_items):
        # find matching entry in results
        keys = list(results.keys())
        candidates = []
        for k in keys:
            if isinstance(k, tuple) and len(k) >= 1 and k[0] == dset_name:
                candidates.append(k)
            elif k == dset_name:
                candidates.append(k)

        if not candidates:
            raise KeyError(f"results dict has no entry for dataset {dset_name}")
        if len(candidates) > 1:
            raise ValueError(
                f"Multiple results entries found for dataset {dset_name}: {candidates}"
            )

        res = results[candidates[0]]

        xs, ys = res["LinePlot"]
        xs = jnp.asarray(xs)
        ys = jnp.asarray(ys)

        ys_min = float(jnp.min(ys))
        ys_max = float(jnp.max(ys))
        denom = ys_max - ys_min
        if abs(denom) < 1e-12:
            ys_norm = jnp.zeros_like(ys)
        else:
            ys_norm = (ys - ys_min) / denom

        xs_np = np.asarray(xs)
        ys_norm_np = np.asarray(ys_norm)

        ax.plot(xs_np, ys_norm_np, linewidth=2.0, color="black", alpha=0.8)
        # segments_lp = np.stack([np.column_stack([xs_np[:-1], ys_norm_np[:-1]]), np.column_stack([xs_np[1:], ys_norm_np[1:]])], axis=1)
        # lc_lp = LineCollection(
        #     segments_lp,
        #     array=ys_norm_np[:-1],
        #     cmap="RdBu_r",
        #     norm=plt.Normalize(vmin=float(ys_norm_np.min()), vmax=float(ys_norm_np.max())),
        #     linewidth=2.0,
        #     zorder=1,
        # )
        # ax.add_collection(lc_lp)

        for key_name, label in sorted(methods):
            dct = res[key_name]
            w = float(dct["weights"][0])
            idx = int(np.argmin(np.abs(xs_np - w)))
            style = marker_styles[key_name]
            sc = ax.scatter(xs_np[idx], ys_norm_np[idx], label=label, **style)

            if label not in legend_labels:
                legend_labels.append(label)
                legend_handles.append(sc)

        # ax.set_title(dset_name)

    for ax in axes_arr:
        ax.set_xlabel("group weight 1")
    axes_arr[0].set_ylabel("fairness on\nPareto front")

    return legend_handles, legend_labels


def plot_two_rows(
    data_dict_row1,
    results_row2,
    dsets_row2,
    alpha=0.9,
    figsize=(8.5, 4.0),
    height_ratios=(1.0, 0.5),
    hspace=0.03,
):
    with plt.rc_context(figsizes.jmlr2001(nrows=2, ncols=3)):
        plt.rcParams.update(tp_axes.tick_direction(x="out", y="out"))
        plt.rcParams["font.family"] = "Open Sans"
        plt.rcParams["font.weight"] = "light"
        plt.rcParams["font.size"] = 9.95
        plt.rcParams["axes.facecolor"] = "white"
        plt.rcParams["figure.dpi"] = 200

        fig, axes = plt.subplots(
            nrows=2,
            ncols=3,
            figsize=figsize,
            gridspec_kw={
                "height_ratios": list(height_ratios),
                "hspace": hspace,
            },
        )
        axes = np.asarray(axes)

        # top row
        cmap, norm, legend_handles1, legend_labels1 = _plot_row_one_on_axes(
            axes[0, :],
            data_dict_row1,
            alpha=alpha,
            cmap="RdBu_r",
            norm=None,
        )

        # bottom row
        legend_handles2, legend_labels2 = _plot_row_two_on_axes(
            axes[1, :],
            results_row2,
            dsets_row2,
            alpha=alpha,
        )

        # shared colorbar for row 1
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=axes[0, :], orientation="vertical", pad=0.02)
        cbar.set_label("Normalized fairness metric")

        # merged legend
        handles_by_label = {}
        for h, lab in zip(legend_handles1, legend_labels1):
            handles_by_label[lab] = h
        for h, lab in zip(legend_handles2, legend_labels2):
            handles_by_label.setdefault(lab, h)

        merged_labels = list(handles_by_label.keys())
        merged_handles = [handles_by_label[l] for l in merged_labels]

        if merged_handles:
            fig.legend(
                merged_handles,
                merged_labels,
                loc="lower center",
                ncol=len(merged_labels),
                frameon=True,
                bbox_to_anchor=(0.5, -0.1),
            )
        plt.savefig("../../figures/figure_2.pdf", bbox_inches="tight")
        # plt.show()
        return fig, axes


def main():
    dsets = load_datasets()
    models = init_models(dsets.keys(), l2_reg=1e-2)
    metrics = init_metrics(models)

    script_dir = Path(__file__).parent
    data_path = script_dir / "data_dict.json"
    results_path = script_dir / "results.json"

    if data_path.exists():
        data_dict = _load_json(data_path)
    else:
        data_dict = build_data_for_datasets(dsets, models, metrics)
        _save_json(data_dict, data_path)

    if results_path.exists():
        results = _load_json(results_path)
    else:
        results = build_results(dsets, models, metrics)
        _save_json(results, results_path)

    plot_two_rows(data_dict, results, dsets)


if __name__ == "__main__":
    main()
