from datetime import datetime
from inspect import currentframe

import jax.numpy as jnp
from scipy.optimize import BFGS, minimize

from .__base__ import Algorithm


class TrustConstr(Algorithm):
    """
    SciPy trust-constr wrapper for simplex-constrained optimization.

    Solves ``min_x oracle.fun(x)`` with equality constraint ``sum(x) = 1 + eps``
    and box bounds using SciPy's ``trust-constr`` method and BFGS Hessian
    approximation.

    Parameters
    ----------
    starting_point : jax.numpy.ndarray or None, optional
        Initial point. If None, uses the uniform distribution.

    Attributes
    ----------
    success : bool or None
        Whether SciPy reported success.
    message : str or None
        Solver message.
    """

    def __init__(self, starting_point=None):
        super().__init__("Trust-Constr")
        if starting_point is None:
            self.starting_point = jnp.full(self.n_groups, 1 / self.n_groups)
        else:
            self.starting_point = starting_point
        self.group_weights = jnp.zeros(self.n_groups)
        self.eps = 1e-12
        self.success = None
        self.message = None

    def run(self, max_iter: int = 500, verbose: int = 1, trace: bool = False):
        """
        Run trust-constr.

        Parameters
        ----------
        max_iter : int, default=500
            Maximum iterations passed to SciPy.
        verbose : int, default=1
            If > 0, prints success and message.
        trace : bool, default=False
            If True, records elapsed time and objective values via a callback.

        Returns
        -------
        None
            This method sets ``self.group_weights`` in-place.
        """
        if trace:
            self.start = datetime.now()

            def callback(x, state):
                delta = (datetime.now() - self.start).total_seconds()
                self.history_time.append(delta)
                self.history_f.append(currentframe().f_back.f_locals["res"]["fun"])  # type: ignore

        def objective(x):
            if self.oracle is not None:
                return self.oracle.fun(x)

        def grad(x):
            if self.oracle is not None:
                return self.oracle.grad(x)

        def constraint_eq(x):
            return jnp.sum(x) - 1.0 - self.eps

        bounds = [(0 + self.eps, 1 + self.eps)] * self.starting_point.shape[0]
        constraints = {"type": "eq", "fun": constraint_eq}

        if trace:
            result = minimize(
                objective,
                self.starting_point,
                method="trust-constr",
                jac=grad,
                hess=BFGS(),
                bounds=bounds,
                constraints=constraints,
                options={"gtol": 1e-5, "disp": verbose, "maxiter": 500},
                callback=callback,  # type: ignore
            )
        else:
            result = minimize(
                objective,
                self.starting_point,
                method="trust-constr",
                jac=grad,
                hess=BFGS(),
                bounds=bounds,
                constraints=constraints,
                options={"gtol": 1e-5, "disp": verbose, "maxiter": max_iter},
            )

        if verbose:
            print(f"[TRUST-CONSTR] Success: {result.success}")
            print(f"[TRUST-CONSTR] Message: {result.message}")
        self.success = result.success
        self.message = result.message
        self.group_weights = result.x
