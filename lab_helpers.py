"""Helper utilities for the BADR fairness lab.

This module provides the *given* infrastructure for the two lab notebooks
(``fairness_badr_lab.ipynb`` and ``fairness_badr_lab_solutions.ipynb``):

* data / model glue around the :mod:`badr` toolbox
  (:func:`make_model`, :func:`group_lists`, :func:`fit_weights`,
  :func:`uniform_weights`, :func:`balanced_weights`);
* small checking / reporting helpers
  (:func:`check_close`, :func:`print_weighting_table`, :func:`minmax_reference`);
* the BADR-GD driver that consumes the student-written derivative oracles
  (:func:`run_badr`);
* the three-group simplex visualization
  (:func:`default_markers`, :func:`plot_simplex_figure`).

Students only fill in the mathematical pieces inside the notebooks; everything
here is meant to be used as-is.
"""

import jax
from jax import config

config.update("jax_enable_x64", True)  # the lab works in float64

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.tri as mtri  # noqa: E402
import matplotlib.patheffects as pe  # noqa: E402

import badr  # noqa: E402


# --------------------------------------------------------------------------- #
# 1. Data and model glue
# --------------------------------------------------------------------------- #
def make_model(l2_reg=1e-1):
    """Build the lower-level model: intercept-free group-weighted logistic regression.

    The parameter vector ``w`` is exactly the linear coefficients, so the score
    of an example ``x`` is ``x @ w``.

    Parameters
    ----------
    l2_reg : float, default=1e-1
        L2 regularization strength ``mu``.

    Returns
    -------
    badr.models.LogisticRegression
        Unfitted model with ``fit_intercept=False``.
    """
    return badr.models.LogisticRegression(l2_reg=l2_reg, fit_intercept=False)


def group_lists(dset, train_test="train"):
    """Return per-group ``(X_list, y_list)`` as float64 JAX arrays.

    Parameters
    ----------
    dset : badr.datasets.Dataset
        Dataset to read from.
    train_test : {"train", "test"}, default="train"
        Split to extract.

    Returns
    -------
    (list[jax.numpy.ndarray], list[jax.numpy.ndarray])
        ``X_list[s]`` has shape ``(n_s, d)`` and ``y_list[s]`` shape ``(n_s,)``.
    """
    if train_test == "train":
        Xs, ys = dset.X_train_list, dset.y_train_list
    else:
        Xs, ys = dset.X_test_list, dset.y_test_list
    X_list = [jnp.asarray(X, dtype=jnp.float64) for X in Xs]
    y_list = [jnp.asarray(y, dtype=jnp.float64) for y in ys]
    return X_list, y_list


def uniform_weights(dset):
    """ERM-style group weighting ``lambda_s = 1 / S``."""
    S = dset.n_groups
    return jnp.ones(S, dtype=jnp.float64) / S


def balanced_weights(dset):
    """Inverse group-size weighting ``lambda_s proportional to 1 / n_s``."""
    sizes = jnp.array([X.shape[0] for X in dset.X_train_list], dtype=jnp.float64)
    inv = 1.0 / sizes
    return inv / jnp.sum(inv)


def fit_weights(model, dset, lam):
    """Train ``w^*(lambda)`` and return the fitted coefficient vector.

    Sets the per-group weights on ``model``, fits it on the training split, and
    returns the coefficients ``w`` (shape ``(d,)``, float64 JAX array).

    Parameters
    ----------
    model : badr.models.LogisticRegression
        Lower-level model (mutated in place by the fit).
    dset : badr.datasets.Dataset
        Dataset providing the training arrays and group indices.
    lam : array-like, shape (S,)
        Group weights on the simplex.

    Returns
    -------
    jax.numpy.ndarray, shape (d,)
        Fitted coefficients ``w^*(lambda)``.
    """
    lam = np.asarray(lam, dtype=float)
    model.set_group_weights(lam)
    model.fit(dset.X_train, dset.y_train, dset.groups)
    return jnp.asarray(model.coef_, dtype=jnp.float64)


def _group_losses(model, X_list, y_list, w):
    """Per-group regularized losses ``ell_s(w)`` as a length-S JAX array."""
    return jnp.array([model._loss(w, X, y) for X, y in zip(X_list, y_list)])


# --------------------------------------------------------------------------- #
# 2. Checking and reporting helpers
# --------------------------------------------------------------------------- #
def check_close(name, value, reference, rtol=1e-5, atol=1e-6):
    """Compare a student value against a reference and print a PASS/FAIL line.

    Parameters
    ----------
    name : str
        Label for the printed line.
    value, reference : array-like
        Student output and the reference value (scalars or arrays).
    rtol, atol : float
        Tolerances forwarded to :func:`numpy.allclose`.

    Returns
    -------
    bool
        Whether the values agree within tolerance.
    """
    a = np.asarray(value, dtype=float)
    b = np.asarray(reference, dtype=float)
    ok = np.allclose(a, b, rtol=rtol, atol=atol)
    max_diff = float(np.max(np.abs(a - b))) if a.size else 0.0
    status = "PASS" if ok else "FAIL"
    print(
        f"[{status}] {name:<22s} yours={np.round(a, 6)}  "
        f"ref={np.round(b, 6)}  max|delta|={max_diff:.2e}"
    )
    return ok


def print_weighting_table(model, dset, metrics, weightings):
    """Fit ``w^*(lambda)`` for each weighting and print a fairness/loss table.

    For every candidate weighting the model is retrained and each fairness
    metric in ``metrics`` is evaluated, alongside the worst-group training loss
    ``max_s ell_s(w)``.

    Parameters
    ----------
    model : badr.models.LogisticRegression
        Lower-level model.
    dset : badr.datasets.Dataset
        Dataset to fit on.
    metrics : dict[str, callable]
        Mapping ``name -> metric(w, X_list, y_list)``.
    weightings : dict[str, array-like]
        Mapping ``name -> lambda`` (each on the simplex).
    """
    X_list, y_list = group_lists(dset, "train")
    metric_names = list(metrics.keys())

    columns = ["weighting"] + metric_names + ["worst-group loss"]
    widths = [max(len(c), 16) for c in columns]
    name_col = max([len(c) for c in weightings] + [len("weighting")])
    widths[0] = name_col

    def fmt_row(cells):
        return "  ".join(f"{c:>{w}}" for c, w in zip(cells, widths))

    print(fmt_row(columns))
    print("  ".join("-" * w for w in widths))
    for wname, lam in weightings.items():
        w = fit_weights(model, dset, lam)
        cells = [wname]
        for mname in metric_names:
            cells.append(f"{float(metrics[mname](w, X_list, y_list)):.4f}")
        worst = float(jnp.max(_group_losses(model, X_list, y_list, w)))
        cells.append(f"{worst:.4f}")
        print(fmt_row(cells))


def minmax_reference(dset, l2_reg=1e-1):
    """Reference min-max (worst-group) logistic-regression solution.

    Uses :class:`badr.models.NonsmoothMinMaxLogisticRegression` to solve the
    epigraph program and recover the normalized per-group dual weights.

    Parameters
    ----------
    dset : badr.datasets.Dataset
        Dataset providing the per-group training arrays.
    l2_reg : float, default=1e-1
        L2 regularization strength.

    Returns
    -------
    (numpy.ndarray, numpy.ndarray)
        ``(coef, group_weights)`` -- the fitted coefficients and the normalized
        dual weighting on the simplex.
    """
    clf = badr.models.NonsmoothMinMaxLogisticRegression(l2_reg=l2_reg, solver="SCS")
    clf.fit(dset.X_train_list, dset.y_train_list)
    return np.asarray(clf.coef_, dtype=float), np.asarray(
        clf.group_weights_, dtype=float
    )


# --------------------------------------------------------------------------- #
# 3. BADR-GD driver
# --------------------------------------------------------------------------- #
class _Parts:
    """Compiled JAX derivative functions passed to the student oracle functions.

    Attributes
    ----------
    grad_loss(w, X, y)
        ``grad_w ell(w; X, y)``.
    hess_loss(w, X, y)
        ``hess_w ell(w; X, y)``.
    grad_metric(w)
        ``grad_w metric(w)``.
    """

    def __init__(self, model, metric_fn, X_list, y_list):
        self.grad_loss = jax.jit(jax.grad(model._loss))
        self.hess_loss = jax.jit(jax.hessian(model._loss))
        self.grad_metric = jax.jit(jax.grad(lambda w: metric_fn(w, X_list, y_list)))


def _project_simplex(v, z=1.0):
    """Euclidean projection of ``v`` onto ``{x >= 0, sum x = z}``."""
    v = np.asarray(v, dtype=float)
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u) - z
    ind = np.arange(1, len(v) + 1)
    rho = np.nonzero(u - cssv / ind > 0)[0][-1]
    theta = cssv[rho] / (rho + 1)
    return jnp.asarray(np.maximum(v - theta, 0.0))


def run_badr(
    dset,
    model,
    metric_fn,
    oracle_fns,
    w0,
    name="metric",
    max_iter=800,
    step_w=0.5,
    step_v=0.5,
    step_lambda=0.02,
    clip=1.0,
    verbose=False,
):
    """Run single-loop BADR-GD to minimize ``V(lambda) = metric(w^*(lambda))``.

    The driver owns the iteration and the simplex projection; the four
    student-provided callables in ``oracle_fns`` supply the derivative oracles.
    The update is::

        w      <- w - step_w     * grad_w f(w, lambda)
        v      <- v - step_v     * ( grad_w metric(w) + Hess_ww f(w, lambda) v )
        lambda <- Proj_simplex( lambda - step_lambda * clip(J^T v) )

    with ``f(w, lambda) = sum_s lambda_s ell_s(w)`` and
    ``J = d/dlambda grad_w f``.

    Parameters
    ----------
    dset : badr.datasets.Dataset
        Dataset (training split is used).
    model : badr.models.LogisticRegression
        Lower-level model (used only for its loss derivatives).
    metric_fn : callable
        Fairness metric ``metric(w, X_list, y_list)`` to minimize.
    oracle_fns : dict[str, callable]
        Must contain ``"grad_lower_groups"``, ``"grad_upper"``, ``"hvp"`` and
        ``"jt_v"`` with the signatures described in the notebook.
    w0 : array-like, shape (d,)
        Warm-start parameters (typically the ERM solution).
    name : str, default="metric"
        Label used in the verbose printout.
    max_iter : int, default=800
        Number of BADR-GD iterations.
    step_w, step_v : float, default=0.5
        Step sizes for the ``w`` and ``v`` updates.
    step_lambda : float, default=0.02
        Step size for the ``lambda`` update.
    clip : float, default=1.0
        L2-norm clipping threshold for the hypergradient.
    verbose : bool, default=False
        If True, print the recovered weighting.

    Returns
    -------
    numpy.ndarray, shape (S,)
        The learned group weighting ``lambda``.
    """
    X_list, y_list = group_lists(dset, "train")
    parts = _Parts(model, metric_fn, X_list, y_list)
    S = dset.n_groups

    grad_lower_groups = oracle_fns["grad_lower_groups"]
    grad_upper = oracle_fns["grad_upper"]
    hvp = oracle_fns["hvp"]
    jt_v = oracle_fns["jt_v"]

    w = jnp.asarray(w0, dtype=jnp.float64)
    v = jnp.zeros_like(w)
    lam = jnp.ones(S, dtype=jnp.float64) / S

    for _ in range(max_iter):
        grad_groups = grad_lower_groups(parts, w, X_list, y_list)  # (S, d)
        grad_w = grad_groups.T @ lam  # grad_w f(w, lambda)
        grad_metric_w, grad_metric_lam = grad_upper(parts, w, lam)
        Hv = hvp(parts, w, lam, v, X_list, y_list)  # Hess_ww f @ v
        hyper = jt_v(parts, w, v, X_list, y_list) + grad_metric_lam
        hyper = jnp.minimum(1.0, clip / (jnp.linalg.norm(hyper) + 1e-12)) * hyper

        w = w - step_w * grad_w
        v = v - step_v * (grad_metric_w + Hv)
        lam = _project_simplex(lam - step_lambda * hyper)

    lam = np.asarray(lam, dtype=float)
    if verbose:
        print(f"   [{name}] lambda* = {np.round(lam, 3)}")
    return lam


# --------------------------------------------------------------------------- #
# 4. Three-group simplex visualization
# --------------------------------------------------------------------------- #
_MARKER_STYLE = {
    "Uniform": dict(marker="v", color="#662C91", size=240),
    "Balanced": dict(marker="^", color="#DA9F93", size=240),
    "Min-max": dict(marker="s", color="#1F77B4", size=240),
    "BADR": dict(marker="X", color="#C81919", size=340),
}


def default_markers(named_lams):
    """Attach default plotting styles to a ``name -> lambda`` mapping.

    Parameters
    ----------
    named_lams : dict[str, array-like]
        Mapping from label to a simplex weight vector.

    Returns
    -------
    list[dict]
        One dict per marker with keys ``label``, ``lam``, ``marker``,
        ``color`` and ``size``, ready for :func:`plot_simplex_figure`.
    """
    markers = []
    for i, (label, lam) in enumerate(named_lams.items()):
        style = _MARKER_STYLE.get(label, dict(marker="o", color=f"C{i}", size=240))
        markers.append({"label": label, "lam": np.asarray(lam, dtype=float), **style})
    return markers


def _simplex_geometry():
    """Geometry helpers for barycentric <-> 2-D conversion (centered equilateral)."""
    v1 = np.array([1, 0]) - (1 / 3) * np.ones(2)
    v2 = np.array([0, 1]) - (1 / 3) * np.ones(2)
    inv_basis_change = np.linalg.inv(np.array([v1, v2]).T)
    c1 = np.array(
        [np.cos(np.pi / 2 - 2 * np.pi / 3), np.sin(np.pi / 2 - 2 * np.pi / 3)]
    )
    c2 = np.array([0.0, 1.0])
    c3 = np.array(
        [np.cos(np.pi / 2 + 2 * np.pi / 3), np.sin(np.pi / 2 + 2 * np.pi / 3)]
    )
    return {
        "inv_basis_change": inv_basis_change,
        "c1": c1,
        "c2": c2,
        "c3": c3,
        "corners": np.array([c1, c2, c3]),
    }


_GEOM = _simplex_geometry()


def _from_3d_to_2d(u3, geom=_GEOM):
    """Barycentric weights (sum to 1) -> 2-D coords in the centered triangle."""
    restricted = np.asarray(u3, float)[:-1] - (1 / 3) * np.ones(2)
    coord_2d = geom["inv_basis_change"] @ restricted
    return coord_2d[0] * geom["c1"] + coord_2d[1] * geom["c2"]


def simplex_grid(n=45):
    """A barycentric grid over the three-group simplex."""
    ts = np.linspace(0, 1, n)
    b = [[a, c, max(1 - a - c, 0.0)] for a in ts for c in ts if 1 - a - c >= -1e-9]
    b = np.array(b)
    return b / b.sum(1, keepdims=True)


def _plot_value_simplex(
    ax, metric_fn, X_list, y_list, grid, grid_coefs, markers, title
):
    """Color one simplex panel by ``V(lambda)`` and draw the markers."""
    Z = np.array([float(metric_fn(c, X_list, y_list)) for c in grid_coefs])
    lo, hi = Z.min(), Z.max()
    Zn = np.clip((Z - lo) / (hi - lo + 1e-12), 0, 1)  # 0 = fairest, 1 = least fair

    xy = np.array([_from_3d_to_2d(lam) for lam in grid])
    tri = mtri.Triangulation(xy[:, 0], xy[:, 1])
    cf = ax.tripcolor(tri, Zn, cmap="Blues", vmin=0, vmax=1)
    ax.tricontour(tri, Zn, levels=15, colors="k", linewidths=0.3)

    for mk in markers:
        h = ax.scatter(
            *_from_3d_to_2d(mk["lam"]),
            marker=mk["marker"],
            s=mk["size"],
            color=mk["color"],
            alpha=0.85,
            zorder=8,
            label=mk["label"],
        )
        h.set_path_effects([pe.withStroke(linewidth=2.0, foreground="black")])

    corners = _GEOM["corners"]
    ax.add_patch(
        plt.Polygon(corners, closed=True, fill=False, edgecolor="k", lw=1.3, zorder=4)
    )
    ax.text(corners[0, 0] + 0.06, corners[0, 1] - 0.04, "$e_1$", ha="left", va="top")
    ax.text(corners[1, 0], corners[1, 1] + 0.06, "$e_2$", ha="center", va="bottom")
    ax.text(corners[2, 0] - 0.06, corners[2, 1] - 0.04, "$e_3$", ha="right", va="top")
    pad = 0.105
    ax.set_xlim(corners[:, 0].min() - pad, corners[:, 0].max() + pad)
    ax.set_ylim(corners[:, 1].min() - pad, corners[:, 1].max() + pad)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, pad=10)
    return cf


def plot_simplex_figure(dset, model, panels, grid_n=45, title=None):
    """Plot one simplex panel per fairness metric.

    Each panel colors the three-group weight simplex by the outer objective
    ``V(lambda) = metric(w^*(lambda))`` (lighter = lower = fairer) and overlays
    the supplied markers. The lower-level fits over the grid are shared across
    panels.

    Parameters
    ----------
    dset : badr.datasets.Dataset
        Dataset (training split is used).
    model : badr.models.LogisticRegression
        Lower-level model used to fit ``w^*(lambda)`` at each grid point.
    panels : list[tuple]
        Each entry is ``(name, metric_fn, markers)`` where ``markers`` comes
        from :func:`default_markers`.
    grid_n : int, default=45
        Resolution of the barycentric grid.
    title : str, optional
        Figure suptitle.

    Returns
    -------
    matplotlib.figure.Figure
        The assembled figure.
    """
    X_list, y_list = group_lists(dset, "train")

    grid = simplex_grid(grid_n)
    grid_coefs = [fit_weights(model, dset, lam) for lam in grid]

    n_panels = len(panels)
    fig, axes = plt.subplots(
        1, n_panels, figsize=(4.3 * n_panels, 4.7), constrained_layout=True
    )
    if n_panels == 1:
        axes = [axes]

    cf = None
    for ax, (name, metric_fn, markers) in zip(axes, panels):
        cf = _plot_value_simplex(
            ax, metric_fn, X_list, y_list, grid, grid_coefs, markers, name
        )

    cbar = fig.colorbar(cf, ax=list(axes), fraction=0.03, pad=0.02)
    cbar.set_label("fairness value  (lower = fairer)")
    cbar.ax.tick_params(labelsize=8)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), frameon=False)

    if title:
        fig.suptitle(title)
    return fig
