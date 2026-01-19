from typing import List

import jax.numpy as jnp
import numpy as np
import cvxpy as cp
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.svm import LinearSVC
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

from badr.models import Model


class SVM(BaseEstimator, ClassifierMixin, Model):
    """
    Group-weighted linear SVM (sklearn ``LinearSVC``).

    Parameters
    ----------
    group_weights : numpy.ndarray, default=np.array([])
        Group weights. If empty, uses uniform over groups.
    l2_reg : float, default=1e-3
        L2 regularization strength (mapped to ``C = 1 / l2_reg``).
    fit_intercept : bool, default=True
        Whether to fit an intercept.
    tol : float, default=1e-9
        Solver tolerance.
    max_iter : int, default=1000
        Maximum iterations.
    random_state : int, default=42
        Passed to sklearn.
    """

    def __init__(
        self,
        group_weights: np.ndarray = np.array([]),
        l2_reg: float = 1e-3,  # constant ridge strength
        fit_intercept: bool = True,
        tol: float = 1e-9,
        max_iter: int = 1000,
        random_state: int = 42,
    ) -> None:
        self.group_weights = group_weights
        self.l2_reg = float(l2_reg)
        self.fit_intercept = fit_intercept
        self.tol = tol
        self.max_iter = max_iter
        self.random_state = random_state

    # Optional analytic objective for diagnostics (not used by LinearSVC)
    def _loss(self, w: np.ndarray, X: np.ndarray, y: np.ndarray) -> float:
        X_j = jnp.asarray(X, dtype=jnp.float64)
        w_j = jnp.asarray(w, dtype=jnp.float64)
        y_j = jnp.asarray(y, dtype=jnp.float64)
        y_signed = 2 * y_j - 1
        if self.fit_intercept and w_j.shape[0] == X_j.shape[1] + 1:
            intercept = w_j[0]
            coef = w_j[1:]
        else:
            intercept = 0.0
            coef = w_j
        margin = y_signed * (X_j @ coef + intercept)
        loss = jnp.mean(jnp.square(jnp.maximum(0.0, 1.0 - margin)))
        reg = 0.5 * self.l2_reg * jnp.dot(coef, coef)  # constant reg (no /N)
        return float(loss + reg)

    def fit(self, X: np.ndarray, y: np.ndarray, groups: List[np.ndarray]):
        # Build group lists
        X_list = [X[g] for g in groups]
        y_list = [y[g] for g in groups]
        if len(X_list) != len(y_list):
            raise ValueError("len(X_list) must equal len(y_list)")
        G = len(X_list)

        # Group weights (default uniform)
        gw = (
            np.asarray(self.group_weights, dtype=float)
            if self.group_weights.size != 0
            else np.full(G, 1.0 / G)
        )
        if gw.shape != (G,):
            raise ValueError(f"group_weights must have shape ({G},), got {gw.shape}")
        self._gw = np.array(gw, dtype=np.float64)

        # Validate and stack
        X_parts, y_parts, group_ns = [], [], []
        for Xi, yi in zip(X_list, y_list):
            Xi_arr = check_array(Xi, ensure_2d=True)
            yi_arr = check_array(yi, ensure_2d=False)
            Xi_arr, yi_arr = check_X_y(Xi_arr, yi_arr)
            group_ns.append(Xi_arr.shape[0])
            X_parts.append(np.asarray(Xi_arr, dtype=np.float64))
            y_parts.append(np.asarray(yi_arr, dtype=np.float64))

        d = X_parts[0].shape[1]
        for Xi in X_parts:
            if Xi.shape[1] != d:
                raise ValueError("All groups must have the same n_features")

        self.n_features_in_ = d
        self.classes_ = np.array([0, 1])

        X_flat = np.vstack(X_parts)
        y_flat = np.concatenate(y_parts)

        # Sample weights so that each group contributes gw[g] * mean_loss_g
        sample_weight = np.zeros_like(y_flat, dtype=np.float64)
        start = 0
        for weight, n_i in zip(self._gw, group_ns):
            end = start + n_i
            sample_weight[start:end] = weight / n_i
            start = end
        # Sum of weights is 1 → matches mean-loss scaling with C = 1/l2_reg

        if self.l2_reg <= 0:
            raise ValueError("l2_reg must be positive when using LinearSVC")

        clf = LinearSVC(
            penalty="l2",
            loss="squared_hinge",
            dual=False,
            C=float(1.0 / self.l2_reg),  # constant, independent of N
            fit_intercept=self.fit_intercept,
            tol=self.tol,
            max_iter=self.max_iter,
            random_state=self.random_state,
        )
        clf.fit(X_flat, y_flat, sample_weight=sample_weight)

        self._clf = clf
        self._sample_weight = sample_weight
        self.coef_ = clf.coef_.ravel().astype(np.float64)
        self.intercept_ = float(clf.intercept_[0]) if self.fit_intercept else 0.0
        return self

    def _group_loss(
        self, dset, train_test: str = "train", w: np.ndarray = np.array([])
    ) -> jnp.ndarray:
        """
        Return per-group squared-hinge *mean* loss (no reg added).
        """
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
            y_signed = 2 * yi_j - 1
            margin = y_signed * (
                Xi_j @ coef + (intercept if self.fit_intercept else 0.0)
            )
            group_loss = jnp.mean(jnp.square(jnp.maximum(0.0, 1.0 - margin)))
            losses.append(group_loss)
        return jnp.stack(losses)

    def decision_function(self, X):
        check_is_fitted(self, ["coef_", "intercept_"])
        X_arr = check_array(X)
        return X_arr.dot(self.coef_) + self.intercept_

    def predict(self, X):
        scores = self.decision_function(X)
        return self.classes_[(scores >= 0).astype(int)]


class NSMMSVM(BaseEstimator, ClassifierMixin, Model):
    """
    Minimax linear SVM via CVXPY.

    Parameters
    ----------
    l2_reg : float, default=1e-3
        L2 regularization strength.
    fit_intercept : bool, default=True
        Whether to fit an intercept.
    solver : str or None, default="SCS"
        CVXPY solver name.
    """

    def __init__(
        self,
        l2_reg: float = 1e-3,  # constant ridge strength on w (not on intercept)
        fit_intercept: bool = True,
        solver: str
        | None = "SCS",  # "SCS" or "ECOS" are good defaults; choose per problem size
    ) -> None:
        self.l2_reg = float(l2_reg)
        self.fit_intercept = fit_intercept
        self.solver = solver

    def fit(self, X: np.ndarray, y: np.ndarray, groups: List[np.ndarray]):
        # --------- Split & validate groups ----------
        X_list = [X[g] for g in groups]
        y_list = [y[g] for g in groups]
        if len(X_list) != len(y_list):
            raise ValueError("len(X_list) must equal len(y_list)")

        X_parts, y_parts, n_parts = [], [], []
        for Xi, yi in zip(X_list, y_list):
            Xi_arr = check_array(Xi, ensure_2d=True)
            yi_arr = check_array(yi, ensure_2d=False)
            Xi_arr, yi_arr = check_X_y(Xi_arr, yi_arr)
            if Xi_arr.shape[0] == 0:
                raise ValueError("Encountered an empty group.")
            X_parts.append(np.asarray(Xi_arr, dtype=np.float64))
            y_parts.append(np.asarray(yi_arr, dtype=np.float64))
            n_parts.append(Xi_arr.shape[0])

        d = X_parts[0].shape[1]
        for Xi in X_parts:
            if Xi.shape[1] != d:
                raise ValueError("All groups must have the same n_features")

        self.n_features_in_ = d
        self.classes_ = np.array([0, 1])

        if self.l2_reg <= 0:
            raise ValueError("l2_reg must be positive for NonsmoothMinMaxSVM")

        # --------- CVXPY variables ----------
        w = cp.Variable(d)
        b = cp.Variable() if self.fit_intercept else 0.0
        t = cp.Variable()  # epigraph var (the max over groups)
        norm_var = cp.Variable(nonneg=True)  # models ||w||_2 via SOC

        constraints = [cp.SOC(norm_var, w)]
        quad_term = 0.5 * float(self.l2_reg) * cp.square(norm_var)

        # --------- Per-group constraints (squared hinge) ----------
        # For each group g, introduce slack z_g >= max(0, 1 - margin_g)
        # mean_squared_hinge_g = (1/n_g) * sum(z_g^2)
        group_constraints: list[cp.constraints.constraint.Constraint] = []

        for Xi, yi in zip(X_parts, y_parts):
            n_g = Xi.shape[0]
            y_signed = 2.0 * yi - 1.0  # in {-1, +1}
            margin = cp.multiply(y_signed, Xi @ w + (b if self.fit_intercept else 0.0))

            z = cp.Variable(n_g)  # hinge slacks
            constraints += [z >= 0, z >= 1.0 - margin]

            mse_hinge = cp.sum_squares(z) / n_g
            c = mse_hinge + quad_term <= t
            constraints.append(c)
            group_constraints.append(c)

        # --------- Solve ----------
        problem = cp.Problem(cp.Minimize(t), constraints)
        solve_kwargs = {}
        if self.solver is not None:
            solve_kwargs["solver"] = self.solver
        try:
            problem.solve(**solve_kwargs)
        except cp.error.SolverError as err:
            raise RuntimeError("CVXPY failed to solve the min-max SVM problem") from err

        if problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
            raise RuntimeError(f"Optimization failed with status {problem.status}")

        # --------- Store solution & dual-based group weights ----------
        self.coef_ = np.asarray(w.value, dtype=np.float64)
        self.intercept_ = float(b.value) if self.fit_intercept else 0.0
        self.opt_value_ = float(t.value)
        self.problem_status_ = problem.status

        duals = np.array([c.dual_value for c in group_constraints], dtype=np.float64)
        duals = np.clip(duals, a_min=0.0, a_max=None)
        s = float(duals.sum())
        self.group_weights_ = None if s <= 0 else (duals / s)

        return self

    # --------- Inference ----------
    def decision_function(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self, ["coef_", "intercept_"])
        X_arr = check_array(X, ensure_2d=True)
        return X_arr.dot(self.coef_) + self.intercept_

    def predict(self, X: np.ndarray) -> np.ndarray:
        scores = self.decision_function(X)
        # classes_ = [0, 1]; threshold at 0
        return self.classes_[(scores >= 0).astype(int)]

    # (Optional) Per-group squared-hinge means, for analysis parity with your original helper.
    def _per_group_mean_squared_hinge(
        self,
        coef: np.ndarray,
        intercept: float,
        X_list: List[np.ndarray],
        y_list: List[np.ndarray],
    ) -> np.ndarray:
        losses = []
        for Xi, yi in zip(X_list, y_list):
            Xi_arr = np.asarray(Xi, dtype=np.float64)
            yi_arr = np.asarray(yi, dtype=np.float64)
            y_signed = 2 * yi_arr - 1
            margin = y_signed * (
                Xi_arr @ coef + (intercept if self.fit_intercept else 0.0)
            )
            hinge = np.maximum(0.0, 1.0 - margin)
            losses.append(np.mean(hinge * hinge))
        return np.asarray(losses, dtype=np.float64)

    """
    Worst-group (minimax) linear ε-SVR with constant L2 regularization, solved via convex programming.

    Solves:
        minimize_{w, b, t}   t
        s.t.   (1/n_g) * mean_eps_insensitive_g(w, b) + 0.5 * l2_reg * ||w||_2^2 <= t,  for all groups g
               ||w||_2 <= norm_var  (SOC to model the L2 penalty)
               s_g >= 0,  s_g >= |y_g - (X_g w + b)| - epsilon
           where:
               mean_eps_insensitive_g = (1/n_g) * sum(s_g)                  if squared=False
                                      = (1/n_g) * sum( s_g^2 )             if squared=True

    Notes
    -----
    - Targets y are real-valued (regression).
    - Intercept is NOT regularized (ridge on weights only).
    - `group_weights_` is the normalized vector of dual variables for the per-group epigraph constraints.
    """

    def __init__(
        self,
        l2_reg: float = 1e-3,  # constant ridge strength on w (not on intercept)
        epsilon: float = 0.1,  # ε-tube
        squared: bool = False,  # use squared ε-insensitive loss if True
        fit_intercept: bool = True,
        solver: str | None = "SCS",  # "SCS" or "ECOS" are good defaults
    ) -> None:
        self.l2_reg = float(l2_reg)
        self.epsilon = float(epsilon)
        self.squared = bool(squared)
        self.fit_intercept = fit_intercept
        self.solver = solver

    def fit(self, X: np.ndarray, y: np.ndarray, groups: List[np.ndarray]):
        # --------- Split & validate groups ----------
        X_list = [X[g] for g in groups]
        y_list = [y[g] for g in groups]
        if len(X_list) != len(y_list):
            raise ValueError("len(X_list) must equal len(y_list)")

        X_parts, y_parts, n_parts = [], [], []
        for Xi, yi in zip(X_list, y_list):
            Xi_arr = check_array(Xi, ensure_2d=True)
            yi_arr = check_array(yi, ensure_2d=False)
            Xi_arr, yi_arr = check_X_y(Xi_arr, yi_arr, y_numeric=True)
            if Xi_arr.shape[0] == 0:
                raise ValueError("Encountered an empty group.")
            X_parts.append(np.asarray(Xi_arr, dtype=np.float64))
            y_parts.append(np.asarray(yi_arr, dtype=np.float64))
            n_parts.append(Xi_arr.shape[0])

        d = X_parts[0].shape[1]
        for Xi in X_parts:
            if Xi.shape[1] != d:
                raise ValueError("All groups must have the same n_features")

        self.n_features_in_ = d

        if self.l2_reg <= 0:
            raise ValueError("l2_reg must be positive for NSMMSVR")

        # --------- CVXPY variables ----------
        w = cp.Variable(d)
        b = cp.Variable() if self.fit_intercept else 0.0
        t = cp.Variable()  # epigraph var (the max over groups)
        norm_var = cp.Variable(nonneg=True)  # models ||w||_2 via SOC

        constraints = [cp.SOC(norm_var, w)]
        quad_term = 0.5 * float(self.l2_reg) * cp.square(norm_var)

        # --------- Per-group constraints (ε-insensitive) ----------
        # residual r_g = y_g - (X_g w + b)
        # slack s_g >= 0, s_g >= |r_g| - ε
        group_constraints: list[cp.constraints.constraint.Constraint] = []
        eps = float(self.epsilon)

        for Xi, yi in zip(X_parts, y_parts):
            n_g = Xi.shape[0]
            pred = Xi @ w + (b if self.fit_intercept else 0.0)
            resid = yi - pred

            s = cp.Variable(n_g)
            constraints += [s >= 0, s >= cp.abs(resid) - eps]

            if self.squared:
                mean_loss_g = cp.sum_squares(s) / n_g
            else:
                mean_loss_g = cp.sum(s) / n_g

            c = mean_loss_g + quad_term <= t
            constraints.append(c)
            group_constraints.append(c)

        # --------- Solve ----------
        problem = cp.Problem(cp.Minimize(t), constraints)
        solve_kwargs = {}
        if self.solver is not None:
            solve_kwargs["solver"] = self.solver
        try:
            problem.solve(**solve_kwargs)
        except cp.error.SolverError as err:
            raise RuntimeError("CVXPY failed to solve the min-max SVR problem") from err

        if problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
            raise RuntimeError(f"Optimization failed with status {problem.status}")

        # --------- Store solution & dual-based group weights ----------
        self.coef_ = np.asarray(w.value, dtype=np.float64)
        self.intercept_ = float(b.value) if self.fit_intercept else 0.0
        self.opt_value_ = float(t.value)
        self.problem_status_ = problem.status

        duals = np.array([c.dual_value for c in group_constraints], dtype=np.float64)
        duals = np.clip(duals, a_min=0.0, a_max=None)
        s = float(duals.sum())
        self.group_weights_ = None if s <= 0 else (duals / s)

        return self

    # --------- Inference ----------
    def decision_function(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self, ["coef_", "intercept_"])
        X_arr = check_array(X, ensure_2d=True)
        return X_arr.dot(self.coef_) + self.intercept_

    def predict(self, X: np.ndarray) -> np.ndarray:
        # For regression, prediction is the raw score
        return self.decision_function(X)

    # (Optional) Per-group ε-insensitive mean losses, paralleling your helper.
    def _per_group_mean_eps_insensitive(
        self,
        coef: np.ndarray,
        intercept: float,
        X_list: List[np.ndarray],
        y_list: List[np.ndarray],
    ) -> np.ndarray:
        losses = []
        eps = float(self.epsilon)
        for Xi, yi in zip(X_list, y_list):
            Xi_arr = np.asarray(Xi, dtype=np.float64)
            yi_arr = np.asarray(yi, dtype=np.float64)
            resid = yi_arr - (
                Xi_arr @ coef + (intercept if self.fit_intercept else 0.0)
            )
            if self.squared:
                loss = np.mean(np.maximum(0.0, np.abs(resid) - eps) ** 2)
            else:
                loss = np.mean(np.maximum(0.0, np.abs(resid) - eps))
            losses.append(loss)
        return np.asarray(losses, dtype=np.float64)
