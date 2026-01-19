from functools import partial

import jax.numpy as jnp
import numpy as np
from jax import jit
from jax.nn import logsumexp, sigmoid
from jax.typing import ArrayLike

from badr.datasets import Dataset
from badr.metrics import FairnessMetric


class DemographicParity(FairnessMetric):
    """
    Demographic parity score across groups.

    Demographic parity is also known as statistical parity, group fairness, independence, or disparate impact.
    """

    def __init__(
        self,
        rho: float = 1.0,
    ):
        super().__init__("Demographic Parity", rho)

    @partial(jit, static_argnums=(0, 2, 3))
    def fun(
        self, w: jnp.ndarray, dset: Dataset, train_test: str = "train"
    ) -> ArrayLike:
        X_list = dset.X_train_list if train_test == "train" else dset.X_test_list
        y_list = dset.y_train_list if train_test == "train" else dset.y_test_list

        all_y = np.concatenate([np.array(y) for y in y_list])
        unique = np.unique(all_y)
        is_binary = unique.size == 2 and set(unique) <= {0, 1}

        preds_list = [jnp.dot(X, w) for X in X_list]

        if is_binary:
            group_probs = jnp.stack(
                [jnp.mean(sigmoid(self.rho * p)) for p in preds_list]
            )
        else:
            group_probs = jnp.stack([jnp.mean(p) for p in preds_list])

        def smooth_abs(x):
            return jnp.sqrt(x**2 + 1e-6)

        dp_score = (
            logsumexp(self.rho * smooth_abs(group_probs - group_probs.mean()))
            / self.rho
        )
        return dp_score
