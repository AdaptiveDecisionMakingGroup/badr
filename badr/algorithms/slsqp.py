from datetime import datetime
from inspect import currentframe

import jax.numpy as jnp
from warnings import filterwarnings

filterwarnings("ignore", category=RuntimeWarning)
from scipy.optimize import minimize

from .__base__ import Algorithm


class SLSQP(Algorithm):
    """
    SLSQP wrapper for simplex-constrained optimization of group weights.

    Solves ``min_x oracle.fun(x)`` subject to ``sum(x) = 1 + eps`` and box bounds
    ``x_i in [eps, 1+eps]`` using SciPy's SLSQP with user-supplied gradient.

    Parameters
    ----------
    starting_point : jax.numpy.ndarray or None, optional
        Initial point. If None, uses the uniform distribution.
    tol : float, default=1e-10
        Function tolerance passed as ``ftol`` to SciPy.

    Attributes
    ----------
    success : bool
        Whether SciPy reported success.
    message : str or None
        Solver message or exception text.
    """

    def __init__(self, starting_point=None, tol=1e-10):
        super().__init__("SLSQP")
        self.starting_point = starting_point
        self.eps = 1e-8
        self.tol = tol
        self.success = False
        self.message = None
        self.my_iterates = []
        self.my_values = []

    def run(self, max_iter: int = 500, verbose: int = 1, trace: bool = False):
        """
        Run SLSQP.

        Parameters
        ----------
        max_iter : int, default=500
            Maximum iterations passed to SciPy.
        verbose : int, default=1
            If > 0, prints success and message (or interruption notice).
        trace : bool, default=False
            If True, records iterates, times, and objective values via a callback.

        Returns
        -------
        None
            This method sets ``self.group_weights`` in-place.

        Raises
        ------
        Exception
            Re-raises exceptions from SciPy after storing the latest trace iterate
            (when available).
        """
        self.group_weights = jnp.zeros(self.n_groups)
        if self.starting_point is None:
            self.starting_point = jnp.full(self.n_groups, 1.0 / self.n_groups)

        callback = None
        if trace:
            self.start = datetime.now()

            def callback(x):
                delta = (datetime.now() - self.start).total_seconds()
                self.history_time.append(delta)
                self.history_lambda.append(jnp.copy(x))
                self.history_f.append(currentframe().f_back.f_locals["fx"])  # type: ignore

        def objective(x):
            return self.oracle.fun(x)

        def grad(x):
            return self.oracle.grad(x)

        def constraint_eq(x):
            return jnp.sum(x) - 1.0 - self.eps

        bounds = [(0 + self.eps, 1 + self.eps)] * self.starting_point.shape[0]
        constraints = {"type": "eq", "fun": constraint_eq}
        minimize_kwargs = dict(
            fun=objective,
            x0=self.starting_point,
            method="SLSQP",
            jac=grad,
            bounds=bounds,
            constraints=constraints,
            options={"ftol": self.tol, "maxiter": max_iter},
        )
        if trace:
            minimize_kwargs["callback"] = callback  # type: ignore

        try:
            result = minimize(**minimize_kwargs)
        except KeyboardInterrupt:
            if verbose > 0:
                print("[SLSQP] Interrupted by user; partial trace preserved.")
            self.success = False
            self.message = "Interrupted by user"
            if trace and self.history_lambda:
                self.group_weights = jnp.copy(self.history_lambda[-1])
            return
        except Exception as exc:
            if trace and self.history_lambda:
                self.group_weights = jnp.copy(self.history_lambda[-1])
            self.success = False
            self.message = str(exc)
            if verbose > 0:
                print(f"[SLSQP] Exception raised: {exc}")
            raise

        if verbose > 0:
            print(f"[SLSQP] Success: {result.success}")
            print(f"[SLSQP] Message: {result.message}")
        self.success = result.success
        self.group_weights = result.x
        self.message = result.message
