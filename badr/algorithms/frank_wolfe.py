from datetime import datetime
from functools import partial

import jax.numpy as jnp
from jax import jit

from .__base__ import Algorithm


class FrankWolfe(Algorithm):
    """
    Frank--Wolfe on the simplex using a linear minimization oracle.

    Uses the oracle gradient ``g = oracle.grad(x)`` and the simplex LMO that
    returns the vertex at ``argmin_i g_i``. The update uses a diminishing step
    size given in :meth:`_step`. Convergence is checked with the FW gap
    ``g_t = -g^T (x - x_prev)``.

    Parameters
    ----------
    starting_point : jax.numpy.ndarray or None, optional
        Initial point on the simplex. If None, uses the uniform distribution.
    eps : float, default=1e-6
        Stopping threshold on the FW gap.

    Attributes
    ----------
    iterates : list[jax.numpy.ndarray]
        Stored iterates (always filled).
    success : bool
        True if the stopping condition was reached.
    message : str or None
        Status message after :meth:`run`.
    """

    def __init__(self, starting_point=None, eps=1e-6):
        super().__init__("Frank-Wolfe")
        self.oracle = None
        self.eps = eps
        self.starting_point = starting_point
        self.success = False
        self.iterates = []
        self.history_x = []
        self.history_time = []
        self.history_fx = []
        self.message = None

    @staticmethod
    def _lmo(grad: jnp.ndarray) -> jnp.ndarray:
        v = jnp.zeros_like(grad)
        idx = jnp.argmin(grad)
        return v.at[idx].set(1.0)

    @staticmethod
    @partial(jit, static_argnums=(2,))
    def _step(x: jnp.ndarray, g: jnp.ndarray, iteration: int) -> jnp.ndarray:
        v = FrankWolfe._lmo(g)
        d = v - x
        alpha = jnp.minimum(
            1.0,
            (2.0 + jnp.log(iteration + 1)) / (iteration + 2 + jnp.log(iteration + 1)),
        )
        return x + alpha * d

    def run(self, max_iter: int = 300, verbose: int = 1, trace: bool = False):
        """
        Run Frank--Wolfe iterations.

        Parameters
        ----------
        max_iter : int, default=300
            Maximum number of iterations.
        verbose : int, default=1
            If > 0, prints success and message.
        trace : bool, default=False
            If True, records iterates and elapsed time.

        Returns
        -------
        jax.numpy.ndarray
            Final group weights.

        Raises
        ------
        ValueError
            If no oracle has been set.
        """
        self.group_weights = jnp.zeros(self.n_groups)
        if self.starting_point is None:
            self.starting_point = jnp.full(self.n_groups, 1.0 / self.n_groups)
        else:
            self.starting_point = self.starting_point
        if self.oracle is None:
            raise ValueError("Oracle not set. Please set the oracle before running.")
        if trace:
            self.start = datetime.now()

        x = self.starting_point
        self.iterates = [x]

        for it in range(max_iter):
            g = self.oracle.grad(x)

            x = FrankWolfe._step(x, g, it)

            self.iterates.append(x)

            if trace:
                now = datetime.now()
                self.history_time.append((now - self.start).total_seconds())
                self.history_x.append(x)

            g_t = jnp.dot(-g, x - self.iterates[-2])
            if g_t < self.eps:
                self.success = True
                break

        self.group_weights = x
        self.message = f"Converged in {len(self.iterates) - 1} iterations (over {max_iter} iterations)."
        if verbose > 0:
            print(f"[FW] Success: {self.success}")
            print(f"[FW] Message: {self.message}")
        return self.group_weights

    def postprocess(self):
        """
        Populate ``history_f`` from the stored ``history_x``.

        Raises
        ------
        ValueError
            If no oracle has been set.
        """
        if self.oracle is None:
            raise ValueError("Oracle not set. Please set the oracle before running.")
        for x in self.history_x:
            self.history_f.append(self.oracle.fun(x))
