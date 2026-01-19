from functools import partial

import jax.numpy as jnp
from jax import jit
from jax.nn import logsumexp, sigmoid
from jax.typing import ArrayLike

from badr.datasets import Dataset
from badr.metrics import FairnessMetric


class EqualizedOdds(FairnessMetric):
    """
    Equalized odds score based on differences in TPR and FPR across groups.
    """

    def __init__(self, rho: float = 1.0):
        super().__init__("Equalized Odds", rho)

    @partial(jit, static_argnums=(0, 2, 3))
    def fun(
        self, w: jnp.ndarray, dset: Dataset, train_test: str = "train"
    ) -> ArrayLike:
        X_list = dset.X_train_list if train_test == "train" else dset.X_test_list
        y_list = dset.y_train_list if train_test == "train" else dset.y_test_list
        preds_list = [jnp.dot(X, w) for X in X_list]

        tpr_per_group = jnp.stack(
            [
                jnp.sum(sigmoid(self.rho * p) * (y == 1)) / (jnp.sum(y == 1) + 1e-15)
                for p, y in zip(preds_list, y_list)
            ]
        )

        fpr_per_group = jnp.stack(
            [
                # sum of “positives” on the y==0 slice
                jnp.sum(sigmoid(self.rho * p) * (y == 0))
                # normalize by the number of true negatives in that group
                / (jnp.sum(y == 0) + 1e-15)
                for p, y in zip(preds_list, y_list)
            ]
        )

        def soft_max(v):
            return logsumexp(self.rho * v) / self.rho

        def soft_min(v):
            return -logsumexp(-self.rho * v) / self.rho

        tpr_diff = soft_max(tpr_per_group) - soft_min(tpr_per_group)

        fpr_diff = soft_max(fpr_per_group) - soft_min(fpr_per_group)

        return (tpr_diff + fpr_diff) / 2
