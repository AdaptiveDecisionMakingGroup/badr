from functools import partial

import jax.numpy as jnp
from jax import grad, jit
from jax.typing import ArrayLike


class FairnessMetric:
    """
    Base class for fairness metrics.

    Parameters
    ----------
    name : str
        Metric name.
    rho : float, default=0.0
        Smoothing/temperature parameter (used by some metrics).
    """

    def __init__(self, name: str, rho: float = 0.0):
        self.name = name
        self.rho = rho

    def set_model(self, model=None):
        """
        Attach a model to the metric.

        Parameters
        ----------
        model : object, optional
            Model instance (required by metrics that call model methods).

        Returns
        -------
        FairnessMetric
            Self.
        """
        self.model = model
        return self

    @partial(jit, static_argnums=(0, 2, 3))
    def fun(self, w, dset, train_test) -> ArrayLike:
        """
        Compute the metric value.

        Parameters
        ----------
        w : ArrayLike
            Parameter vector.
        dset
            Dataset.
        train_test : {"train", "test"}
            Split to use.

        Returns
        -------
        ArrayLike
            Scalar metric value.

        Raises
        ------
        NotImplementedError
            If not implemented.
        """
        raise NotImplementedError("Must be implemented in subclasses.")

    @partial(jit, static_argnums=(0, 2, 3))
    def grad(self, w, dset, train_test="train") -> jnp.ndarray:
        """
        Gradient of :meth:`fun` w.r.t. ``w``.

        Parameters
        ----------
        w : ArrayLike
            Parameter vector.
        dset
            Dataset.
        train_test : {"train", "test"}, default="train"
            Split to use.

        Returns
        -------
        jax.numpy.ndarray
            Gradient array with the same shape as ``w``.
        """

        def f(w_, data):
            return self.fun(w_, data, train_test)

        return grad(f)(w, dset)
