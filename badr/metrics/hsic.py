from functools import partial

import jax.numpy as jnp
from jax import jit
from jax.typing import ArrayLike

from badr.datasets import Dataset
from badr.metrics import FairnessMetric


class HSIC(FairnessMetric):
    """
    HSIC-like dependence score between group index and predictions.
    """

    def __init__(self):
        super().__init__("HSIC")

    @partial(jit, static_argnums=(0, 2, 3))
    def fun(
        self, w: jnp.ndarray, dset: Dataset, train_test: str = "train"
    ) -> ArrayLike:
        X_list = dset.X_train_list if train_test == "train" else dset.X_test_list
        y_list = dset.y_train_list if train_test == "train" else dset.y_test_list

        s = jnp.concatenate(
            [jnp.full_like(y_ii, fill_value=ii) for (ii, y_ii) in enumerate(y_list)]
        )
        s_mean = jnp.mean(s)

        y_hat = jnp.concatenate([X_ii @ w for X_ii in X_list])
        y_mean = jnp.mean(y_hat)

        s_centered = s - s_mean
        y_centered = y_hat - y_mean
        cov_matrix = jnp.dot(s_centered.T, y_centered) / (s.shape[0] - 1)

        return jnp.linalg.norm(cov_matrix) ** 2
