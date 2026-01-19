from functools import partial

import jax.numpy as jnp
from jax import jit
from jax.typing import ArrayLike

from badr.datasets import Dataset
from badr.metrics import FairnessMetric


class DisparateMistreatment(FairnessMetric):
    """
    Disparate mistreatment score based on squared covariance between group index and prediction.
    """

    def __init__(self):
        super().__init__("Disparate Mistreatment")

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
        y_hat_mean = jnp.mean(y_hat)

        cov = jnp.mean((s - s_mean) * (y_hat - y_hat_mean))
        return cov**2
