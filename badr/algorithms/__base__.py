from datetime import datetime

import jax.numpy as jnp
import numpy as np

from badr.oracles import Oracle


class Trace:
    def __init__(self, f=None, freq=1) -> None:
        self.trace_x = []
        self.trace_time = []
        self.trace_fx = []
        self.start = datetime.now()
        self._counter = 0
        self.freq = int(freq)
        self.f = f

    def __call__(self, dl):
        if self._counter % self.freq == 0:
            if self.f is not None:
                self.trace_fx.append(float(self.f(jnp.array(dl["x"]))))
            else:
                self.trace_x.append(np.copy(dl["x"]))
            delta = (datetime.now() - self.start).total_seconds()
            self.trace_time.append(delta)
        self._counter += 1


class Algorithm:
    """
    Base class for algorithms optimizing group weights.

    Stores common bookkeeping (objective trace, timings, iterates) and a reference
    to an :class:`~badr.oracles.Oracle`.

    Parameters
    ----------
    name : str
        Display name for the algorithm.

    Attributes
    ----------
    oracle : Oracle or None
        Oracle providing ``fun``/``grad`` (and possibly stochastic primitives).
    n_groups : int
        Number of groups (set from the oracle).
    group_weights : jax.numpy.ndarray
        Latest group-weight iterate (typically on the simplex).
    history_f : list[float]
        Traced objective values (when enabled by the algorithm / ``trace`` flag).
    history_time : list[float]
        Elapsed time trace (seconds).
    history_lambda : list[jax.numpy.ndarray]
        Traced group-weight iterates (when enabled).
    """

    def __init__(self, name: str) -> None:
        self.name = name
        self.history_f = []
        self.history_time = []
        self.history_lambda = []
        self._last_callback_time = None
        self.group_weights: jnp.ndarray = jnp.array([])
        self.oracle = None

    def set_oracle(self, oracle: Oracle) -> None:
        # Set the oracle for the algorithm.
        self.oracle = oracle
        self.n_groups = oracle.n_groups

    def run(self, max_iter: int = 1, verbose: int = 1, trace: bool = False):
        """
        Run the algorithm.

        Parameters
        ----------
        max_iter : int, default=1
            Maximum number of iterations.
        verbose : int, default=1
            Verbosity level (interpretation is algorithm-specific).
        trace : bool, default=False
            If True, record per-iteration history (objective/iterates/times).

        Raises
        ------
        NotImplementedError
            If not implemented by a subclass.
        """
        raise NotImplementedError("Algorithm.run() must be implemented in subclasses.")
