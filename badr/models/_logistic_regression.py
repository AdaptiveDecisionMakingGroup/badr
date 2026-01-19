from typing import List

import jax.numpy as jnp
from jax import config
import numpy as np
import cvxpy as cp
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression as SkLogisticRegression
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y
from badr.models import Model

config.update("jax_enable_x64", True)


class LogisticRegression(BaseEstimator, ClassifierMixin, Model):
    """
    Group-weighted logistic regression (sklearn ``LogisticRegression``).

    Parameters
    ----------
    group_weights : numpy.ndarray, default=np.array([])
        Group weights. If empty or invalid, falls back to uniform.
    l2_reg : float, default=1e-3
        L2 regularization strength (mapped to ``C = 1 / l2_reg``).
    fit_intercept : bool, default=True
        Whether to fit an intercept.
    random_state : int or None, default=42
        Passed to sklearn.
    """

    def __init__(
        self,
        group_weights: np.ndarray = np.array([]),
        l2_reg: float = 1e-3,
        fit_intercept: bool = True,
        random_state: int | None = 42,
    ) -> None:
        self.group_weights = group_weights
        self.l2_reg = l2_reg
        self.fit_intercept = fit_intercept
        self.random_state = random_state

    def _loss(self, w: np.ndarray, X: np.ndarray, y: np.ndarray) -> float:
        """Average logistic loss + constant L2: 0.5 * l2_reg * ||coef||^2 (no reg on intercept)."""
        X_j = jnp.asarray(X, dtype=jnp.float64)
        w_j = jnp.asarray(w, dtype=jnp.float64)
        y_j = jnp.asarray(y, dtype=jnp.float64)

        if self.fit_intercept and w_j.shape[0] == X_j.shape[1] + 1:
            intercept = w_j[0]
            coef = w_j[1:]
        else:
            intercept = jnp.array(0.0, dtype=jnp.float64)
            coef = w_j

        logits = X_j @ coef + intercept
        loss = jnp.mean(jnp.logaddexp(0.0, logits) - y_j * logits)
        reg = 0.5 * self.l2_reg * jnp.dot(coef, coef)
        return jnp.asarray(loss + reg)

    def fit(self, X: np.ndarray, y: np.ndarray, groups: List[np.ndarray]):
        """
        Fit the logistic regression model on grouped data with group weights.
        """
        X_list = [X[g] for g in groups]
        y_list = [y[g] for g in groups]
        if len(X_list) != len(y_list):
            raise ValueError(
                f"len(X_list)={len(X_list)} must equal len(y_list)={len(y_list)}"
            )
        G = len(X_list)
        gw = (
            np.asarray(self.group_weights, dtype=float)
            if np.isclose(self.group_weights.sum(), 1.0, atol=1e-5)
            else np.full(G, 1.0 / G)
        )
        if gw.shape != (G,):
            raise ValueError(
                f"group_weights must have shape ({G},), got {tuple(gw.shape)}"
            )
        self._gw = np.array(gw, dtype=np.float64)

        X_parts, y_parts, group_ns = [], [], []
        for Xi, yi in zip(X_list, y_list):
            Xi_arr = check_array(Xi, ensure_2d=True)
            yi_arr = check_array(yi, ensure_2d=False)
            Xi_arr, yi_arr = check_X_y(Xi_arr, yi_arr)
            group_ns.append(Xi_arr.shape[0])
            X_parts.append(np.array(Xi_arr, dtype=np.float64))
            y_parts.append(np.array(yi_arr, dtype=np.float64))

        d = X_parts[0].shape[1]
        for Xi in X_parts:
            if Xi.shape[1] != d:
                raise ValueError("All groups must have same n_features")

        self.n_features_in_ = d
        self.classes_ = np.array([0, 1])

        X_flat = np.vstack(X_parts)
        y_flat = np.concatenate(y_parts)

        sample_weight = np.zeros_like(y_flat, dtype=np.float64)
        start = 0
        for weight, n_i in zip(self._gw, group_ns):
            end = start + n_i
            sample_weight[start:end] = weight / n_i
            start = end
        self._group_ns = group_ns

        if self.l2_reg <= 0:
            raise ValueError("l2_reg must be positive when using LogisticRegression")

        clf = SkLogisticRegression(
            penalty="l2",
            solver="lbfgs",
            fit_intercept=self.fit_intercept,
            C=float(1.0 / self.l2_reg),
            random_state=self.random_state,
            max_iter=200,
        )
        clf.fit(X_flat, y_flat, sample_weight=sample_weight)

        self._clf = clf
        self._sample_weight = sample_weight
        self.coef_ = clf.coef_.ravel().astype(np.float64)
        if self.fit_intercept:
            self.intercept_ = float(clf.intercept_[0])
        else:
            self.intercept_ = 0.0
        return self

    def _group_loss(
        self, dset, train_test: str = "train", w: np.ndarray = np.array([])
    ) -> jnp.ndarray:
        """Per-group negative log-likelihood consistent with the training objective."""
        if w.size == 0:
            check_is_fitted(self, ["coef_", "intercept_"])
            coef = jnp.asarray(self.coef_, dtype=jnp.float64)
            intercept = jnp.asarray(self.intercept_, dtype=jnp.float64)
        else:
            w_arr = jnp.asarray(w, dtype=jnp.float64)
            if self.fit_intercept:
                intercept = w_arr[0]
                coef = w_arr[1:]
            else:
                coef = w_arr
                intercept = jnp.array(0.0, dtype=jnp.float64)

        X_list = dset.X_train_list if train_test == "train" else dset.X_test_list
        y_list = dset.y_train_list if train_test == "train" else dset.y_test_list

        losses = []
        for Xi, yi in zip(X_list, y_list):
            Xi_j = jnp.asarray(Xi, dtype=jnp.float64)
            yi_j = jnp.asarray(yi, dtype=jnp.float64)
            logits = Xi_j @ coef
            if self.fit_intercept:
                logits = logits + intercept
            nll = jnp.mean(jnp.logaddexp(0.0, logits) - yi_j * logits)
            losses.append(nll)
        return jnp.stack(losses)

    def decision_function(self, X):
        check_is_fitted(self, ["coef_", "intercept_"])
        X_arr = check_array(X)
        return X_arr.dot(self.coef_) + self.intercept_

    def predict(self, X):
        scores = self.decision_function(X)
        return self.classes_[(scores >= 0).astype(int)]


class NonsmoothMinMaxLogisticRegression(BaseEstimator, ClassifierMixin, Model):
    """
    Minimax logistic regression via CVXPY.

    Parameters
    ----------
    l2_reg : float or None, default=None
        L2 regularization strength. If None, uses ``1 / n_samples``.
    solver : str or None, default="SCS"
        CVXPY solver name.
    """

    def __init__(self, l2_reg: float | None = None, solver: str | None = "SCS") -> None:
        super().__init__()
        self.l2_reg = l2_reg
        self.solver = solver

    def fit(self, X_list: List[np.ndarray], y_list: List[np.ndarray]):
        """Solve the epigraph formulation with a shared L2 regularizer."""

        if len(X_list) == 0:
            raise ValueError("X_list must contain at least one group")
        if len(X_list) != len(y_list):
            raise ValueError("X_list and y_list must have the same length")

        n_samples = sum(len(y) for y in y_list)
        if n_samples == 0:
            raise ValueError("Total number of samples must be positive")

        if self.l2_reg is None:
            self.l2_reg = 1.0 / n_samples
        if self.l2_reg <= 0:
            raise ValueError("l2_reg must be positive")

        n_features = X_list[0].shape[1]
        for X in X_list:
            if X.shape[1] != n_features:
                raise ValueError("All groups must share the same number of features")

        w = cp.Variable(n_features)
        t = cp.Variable()
        norm_var = cp.Variable(nonneg=True)

        constraints = [cp.SOC(norm_var, w)]
        quad_term = 0.5 * self.l2_reg * cp.square(norm_var)

        group_constraints: list[cp.constraints.constraint.Constraint] = []

        for X_group, y_group in zip(X_list, y_list):
            X_arr = np.asarray(X_group, dtype=np.float64)
            y_arr = np.asarray(y_group, dtype=np.float64)
            if X_arr.shape[0] == 0:
                raise ValueError("Each group must contain at least one sample")
            logits = X_arr @ w
            logistic_loss = (
                cp.sum(cp.logistic(logits) - cp.multiply(y_arr, logits))
                / X_arr.shape[0]
            )
            group_constraint = logistic_loss + quad_term <= t
            constraints.append(group_constraint)
            group_constraints.append(group_constraint)

        objective = cp.Minimize(t)
        problem = cp.Problem(objective, constraints)

        solve_kwargs = {}
        if self.solver is not None:
            solve_kwargs["solver"] = self.solver
        try:
            problem.solve(**solve_kwargs)
        except cp.error.SolverError as err:
            raise RuntimeError("CVXPY failed to solve the min-max problem") from err

        if problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
            raise RuntimeError(f"Optimization failed with status {problem.status}")

        self.coef_ = np.asarray(w.value, dtype=np.float64)
        self.opt_value_ = float(t.value)
        self.problem_status_ = problem.status

        duals = np.array([c.dual_value for c in group_constraints], dtype=np.float64)
        duals = np.clip(duals, a_min=0.0, a_max=None)
        dual_sum = float(duals.sum())
        self.group_weights_ = None if dual_sum <= 0 else (duals / dual_sum)

        return self

    def _predict(self, X: np.ndarray):
        if not hasattr(self, "coef_"):
            raise ValueError("Model has not been fitted yet.")
        return np.asarray(X, dtype=np.float64) @ self.coef_

    def predict(self, X_list: List[np.ndarray]):
        return [self._predict(X) for X in X_list]
