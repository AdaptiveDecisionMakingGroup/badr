from __future__ import annotations

from typing import List

import jax.numpy as jnp
import numpy as np
import cvxpy as cp
from jax import Array
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import Ridge as SkRidge
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

from badr.models import Model


class RidgeRegression(BaseEstimator, RegressorMixin, Model):
    """
    Group-weighted ridge regression (sklearn ``Ridge``).

    Parameters
    ----------
    group_weights : numpy.ndarray, default=np.array([])
        Group weights. If empty, uses uniform over groups.
    l2_reg : float, default=1e-3
        L2 regularization strength (``alpha`` in sklearn).
    fit_intercept : bool, default=True
        Whether to fit an intercept.
    random_state : int or None, default=42
        Kept for API parity with other models.
    """

    def __init__(
        self,
        group_weights: np.ndarray = np.array([]),
        l2_reg: float = 1e-3,
        fit_intercept: bool = True,
        random_state: int | None = 42,  # kept for API parity (unused by Ridge)
    ) -> None:
        self.group_weights = group_weights
        self.l2_reg = float(l2_reg)
        self.fit_intercept = fit_intercept
        self.random_state = random_state

    def fit(
        self, X: np.ndarray, y: np.ndarray, groups: List[np.ndarray]
    ) -> "RidgeRegression":
        # Split into groups
        X_list = [X[g] for g in groups]
        y_list = [y[g] for g in groups]
        if len(X_list) != len(y_list):
            raise ValueError(
                f"len(X_list)={len(X_list)} must equal len(y_list)={len(y_list)}"
            )
        G = len(X_list)

        # Group weights (default uniform over groups)
        gw = (
            np.asarray(self.group_weights, dtype=float)
            if getattr(self.group_weights, "size", 0) != 0
            else np.full(G, 1.0 / G)
        )
        if gw.shape != (G,):
            raise ValueError(
                f"group_weights must have shape ({G},), got {tuple(gw.shape)}"
            )
        self._gw = np.array(gw, dtype=np.float64)

        # Validate groups and collect pieces
        X_parts, y_parts, group_ns = [], [], []
        for Xi, yi in zip(X_list, y_list):
            Xi_np = check_array(Xi, ensure_2d=True)
            yi_np = check_array(yi, ensure_2d=False)
            Xi_np, yi_np = check_X_y(Xi_np, yi_np)
            if Xi_np.shape[0] == 0:
                raise ValueError("Encountered an empty group.")
            X_parts.append(np.asarray(Xi_np, dtype=np.float64))
            y_parts.append(np.asarray(yi_np, dtype=np.float64))
            group_ns.append(Xi_np.shape[0])

        # Consistent feature dimension
        d = X_parts[0].shape[1]
        for Xi in X_parts:
            if Xi.shape[1] != d:
                raise ValueError("All groups must have same n_features")
        self.n_features_in_ = d

        # Flatten
        X_flat = np.vstack(X_parts)
        y_flat = np.concatenate(y_parts)

        # Scale targets for numerical stability
        y_mean = float(np.mean(y_flat))
        y_std = float(np.std(y_flat))
        if y_std < 1e-10:
            y_std = 1.0
        y_scaled = (y_flat - y_mean) / y_std
        self._y_mean = y_mean
        self._y_std = y_std

        # Per-sample weights: gw[g] / n_g for samples in group g
        sample_weight = np.zeros_like(y_flat, dtype=np.float64)
        start = 0
        for w_g, n_g in zip(self._gw, group_ns):
            end = start + n_g
            sample_weight[start:end] = w_g / n_g
            start = end
        self._group_ns = group_ns
        self._sample_weight = sample_weight

        # Sanity
        if self.l2_reg <= 0:
            raise ValueError("l2_reg must be positive for RidgeRegression")

        # Fit via sklearn Ridge (no reg on intercept when fit_intercept=True)
        reg = SkRidge(
            alpha=float(self.l2_reg),
            fit_intercept=self.fit_intercept,
            solver="auto",  # lets sklearn choose a stable solver
            random_state=self.random_state,  # ignored, kept for API parity
        )
        reg.fit(X_flat, y_scaled, sample_weight=sample_weight)

        # Store params in your preferred dtypes
        coef_scaled = np.asarray(reg.coef_, dtype=np.float64)
        intercept_scaled = float(reg.intercept_)
        self.coef_ = coef_scaled * self._y_std
        if self.fit_intercept:
            self.intercept_ = intercept_scaled * self._y_std + self._y_mean
        else:
            self.intercept_ = self._y_mean
        self._reg = reg
        return self

    # === Analysis helpers (kept using JAX for downstream code compatibility) ===

    def _group_loss(
        self,
        dset,
        train_test: str = "train",
        w: np.ndarray | jnp.ndarray = jnp.array([]),
    ) -> Array:
        """
        Per-group MSE consistent with the training objective (no reg term).
        """
        if getattr(w, "size", 0) == 0:
            check_is_fitted(self, ["coef_", "intercept_"])
            coef = jnp.asarray(self.coef_, dtype=jnp.float64)
            intercept = jnp.asarray(self.intercept_, dtype=jnp.float64)
        else:
            w_j = jnp.asarray(w, dtype=jnp.float64)
            if self.fit_intercept:
                intercept = w_j[0]
                coef = w_j[1:]
            else:
                coef = w_j
                intercept = jnp.array(0.0, dtype=jnp.float64)

        X_list = dset.X_train_list if train_test == "train" else dset.X_test_list
        y_list = dset.y_train_list if train_test == "train" else dset.y_test_list

        losses = []
        for Xi, yi in zip(X_list, y_list):
            Xi_j = jnp.asarray(Xi, dtype=jnp.float64)
            yi_j = jnp.asarray(yi, dtype=jnp.float64)
            preds = Xi_j @ coef + (intercept if self.fit_intercept else 0.0)
            losses.append(jnp.mean((preds - yi_j) ** 2))
        return jnp.stack(losses)

    def _loss(self, w: jnp.ndarray, X: np.ndarray, y: np.ndarray) -> Array:
        """
        Scalar loss used for differentiation: MSE + 0.5 * l2_reg * ||w||^2 (no reg on intercept).
        """
        if getattr(w, "size", 0) == 0:
            check_is_fitted(self, ["coef_", "intercept_"])
            coef = jnp.asarray(self.coef_, dtype=jnp.float64)
            intercept = jnp.asarray(self.intercept_, dtype=jnp.float64)
        else:
            w_j = jnp.asarray(w, dtype=jnp.float64)
            if self.fit_intercept:
                intercept = w_j[0]
                coef = w_j[1:]
            else:
                intercept = jnp.array(0.0, dtype=jnp.float64)
                coef = w_j

        X_j = jnp.asarray(check_array(X, ensure_2d=True), dtype=jnp.float64)
        y_j = jnp.asarray(check_array(y, ensure_2d=False), dtype=jnp.float64)
        preds = X_j @ coef + (intercept if self.fit_intercept else 0.0)
        mse = jnp.mean((preds - y_j) ** 2)
        reg = 0.5 * float(self.l2_reg) * jnp.sum(coef**2)
        return mse + reg

    def predict(self, X: np.ndarray) -> jnp.ndarray:
        check_is_fitted(self, ["coef_"])
        X_np = check_array(X, ensure_2d=True)
        X_j = jnp.asarray(X_np, dtype=jnp.float64)
        return jnp.asarray(
            X_j @ jnp.asarray(self.coef_, dtype=jnp.float64) + self.intercept_
        )

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Relative RMSE score: RMSE divided by mean absolute target magnitude.
        """
        check_is_fitted(self, ["coef_"])
        X_np = check_array(X, ensure_2d=True)
        y_np = check_array(y, ensure_2d=False)
        y_pred = self.predict(X_np)
        rmse = np.sqrt(np.mean((y_pred - y_np) ** 2))
        return rmse


class NSMMRR(BaseEstimator, RegressorMixin, Model):
    """
    Minimax ridge regression via CVXPY.

    Parameters
    ----------
    l2_reg : float or None, default=None
        L2 regularization strength. If None, uses ``1 / n_samples``.
    fit_intercept : bool, default=True
        Whether to fit an intercept.
    solver : str or None, default="SCS"
        CVXPY solver name.
    """

    def __init__(
        self,
        l2_reg: float | None = None,
        fit_intercept: bool = True,
        solver: str | None = "SCS",
    ) -> None:
        super().__init__()
        self.l2_reg = l2_reg
        self.fit_intercept = fit_intercept
        self.solver = solver

    def fit(self, X_list, y_list):
        # Validate inputs
        if len(X_list) == 0:
            raise ValueError("X_list must contain at least one group")
        if len(X_list) != len(y_list):
            raise ValueError("X_list and y_list must have the same length")

        # Validate and set l2_reg
        n_samples = sum(len(y) for y in y_list)
        if n_samples == 0:
            raise ValueError("Total number of samples must be positive")

        if self.l2_reg is None:
            self.l2_reg = 1.0 / n_samples
        if self.l2_reg <= 0:
            raise ValueError("l2_reg must be positive for NSMMRR")

        X_list = [np.asarray(Xg, dtype=np.float64) for Xg in X_list]
        y_list = [np.asarray(yg, dtype=np.float64).ravel() for yg in y_list]

        # Scale y for better numerical stability
        y_concat = np.concatenate(y_list)
        y_mean = np.mean(y_concat)
        y_std = np.std(y_concat)
        if y_std < 1e-10:
            y_std = 1.0
        y_list_scaled = [(y - y_mean) / y_std for y in y_list]

        # Validate feature dimensions
        d = X_list[0].shape[1]
        for X in X_list:
            if X.shape[1] != d:
                raise ValueError("All groups must have the same number of features")
            if X.shape[0] == 0:
                raise ValueError("Each group must contain at least one sample")

        self.n_features_in_ = d
        self._y_mean = y_mean
        self._y_std = y_std

        w = cp.Variable(d)
        b = cp.Variable() if self.fit_intercept else 0.0
        t = cp.Variable()
        norm_var = cp.Variable(nonneg=True)  # models ||w||_2 via SOC

        # SOC constraint to model ||w||_2 <= norm_var
        constraints = [cp.SOC(norm_var, w)]
        quad_term = 0.5 * float(self.l2_reg) * cp.square(norm_var)

        group_constraints = []
        for Xg, yg_scaled in zip(X_list, y_list_scaled):
            n_g = Xg.shape[0]
            resid = Xg @ w + (b if self.fit_intercept else 0.0) - yg_scaled
            Fg = cp.sum_squares(resid) / n_g
            c = Fg + quad_term <= t
            constraints.append(c)
            group_constraints.append(c)

        problem = cp.Problem(cp.Minimize(t), constraints)

        solve_kwargs = {}
        if self.solver is not None:
            solve_kwargs["solver"] = self.solver

        # Use relaxed tolerances for better numerical stability
        if self.solver == "SCS" or self.solver is None:
            solve_kwargs.setdefault("eps_abs", 1e-4)
            solve_kwargs.setdefault("eps_rel", 1e-4)
            solve_kwargs.setdefault("max_iters", 10000)

        try:
            problem.solve(**solve_kwargs)
        except cp.error.SolverError as err:
            raise RuntimeError(
                "CVXPY failed to solve the min-max ridge regression problem"
            ) from err

        if problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
            # If the solver reports infeasible_inaccurate, it might still have a usable solution
            # especially for large-scale problems with numerical challenges
            if problem.status == "infeasible_inaccurate" and w.value is not None:
                import warnings

                warnings.warn(
                    f"Solver returned '{problem.status}' but produced a solution. "
                    "Results may be inaccurate. Consider using a different solver or adjusting tolerances.",
                    UserWarning,
                )
            else:
                raise RuntimeError(f"Optimization failed with status {problem.status}")

        self.coef_ = np.asarray(w.value, dtype=np.float64) * self._y_std
        self.intercept_ = (
            float(b.value) if self.fit_intercept else 0.0
        ) * self._y_std + self._y_mean
        self.opt_value_ = float(t.value) * (self._y_std**2)  # Scale back the objective
        self.problem_status_ = problem.status

        # Compute group weights from dual variables
        duals = np.array([c.dual_value for c in group_constraints], dtype=np.float64)
        duals = np.clip(duals, a_min=0.0, a_max=None).ravel()
        dual_sum = float(duals.sum())
        self.group_weights_ = None if dual_sum <= 0 else (duals / dual_sum)

        return self

    # --- API parity helpers ---
    def _predict(self, X: np.ndarray) -> np.ndarray:
        if not hasattr(self, "coef_"):
            raise ValueError("Model has not been fitted yet.")
        X = np.asarray(X, dtype=np.float64)
        return X @ self.coef_ + self.intercept_

    def predict(self, X_list: List[np.ndarray]) -> List[np.ndarray]:
        return [self._predict(X) for X in X_list]

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Relative RMSE score: RMSE divided by mean absolute target magnitude.
        """
        check_is_fitted(self, ["coef_"])
        X_np = check_array(X, ensure_2d=True)
        y_np = check_array(y, ensure_2d=False)
        y_pred = self.predict(X_np)
        rmse = np.sqrt(np.mean((y_pred - y_np) ** 2))
        return rmse
