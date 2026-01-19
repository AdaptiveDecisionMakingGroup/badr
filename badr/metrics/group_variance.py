from functools import partial

import jax.numpy as jnp
from jax import jit
from jax.typing import ArrayLike

from badr.datasets import Dataset
from badr.metrics import FairnessMetric


class GroupVariance(FairnessMetric):
    """
    Variance of per-group losses.

    Requires a model that implements ``_group_loss(dset, train_test, w_full)``.
    """

    def __init__(self, model=None):
        super().__init__("Group Variance")  # links to bounded group loss
        self.model = model

    def set_model(self, model=None):
        if model is None:
            raise ValueError("Model must be provided to compute group variance.")
        self.model = model
        return self

    def _prepare_weights(self, w: jnp.ndarray, dset: Dataset) -> jnp.ndarray:
        fit_intercept = bool(getattr(self.model, "fit_intercept", False))
        n_features = dset.X_train.shape[1]
        expected = n_features + (1 if fit_intercept else 0)

        w = jnp.asarray(w, dtype=jnp.float64)
        if w.shape[0] == expected:
            return w
        if fit_intercept and w.shape[0] == expected - 1:
            intercept = jnp.zeros((1,), dtype=w.dtype)
            return jnp.concatenate([intercept, w], axis=0)
        if not fit_intercept and w.shape[0] == n_features:
            return w
        raise ValueError(
            f"Unexpected weight length {w.shape[0]}; expected {expected}"
            + (" (including intercept)" if fit_intercept else "")
        )

    @partial(jit, static_argnums=(0, 2, 3))
    def _compute_variance(
        self, w_full: jnp.ndarray, dset: Dataset, train_test: str = "train"
    ) -> ArrayLike:
        group_losses = self.model._group_loss(dset, train_test, w_full)
        return jnp.var(group_losses)

    def fun(
        self, w: jnp.ndarray, dset: Dataset, train_test: str = "train"
    ) -> ArrayLike:
        if self.model is None:
            raise ValueError(
                "Model must be provided to compute group variance. Use set_model()."
            )
        w_full = self._prepare_weights(w, dset)
        return self._compute_variance(w_full, dset, train_test)
