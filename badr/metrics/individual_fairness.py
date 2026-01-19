from functools import partial

import jax.numpy as jnp
from jax import jit, vmap
from jax.typing import ArrayLike

from badr.datasets import Dataset
from badr.metrics import FairnessMetric


class IndividualFairness(FairnessMetric):
    """
    Individual fairness score based on cross-group pairwise prediction differences.
    """

    def __init__(self):
        super().__init__("Individual Fairness")

    @partial(jit, static_argnums=(0, 2, 3))
    def fun(
        self, w: jnp.ndarray, dset: Dataset, train_test: str = "train"
    ) -> ArrayLike:
        X_list = dset.X_train_list if train_test == "train" else dset.X_test_list
        y_list = dset.y_train_list if train_test == "train" else dset.y_test_list
        # ensure y values are numeric floats for subtraction
        y_list = [jnp.array(y, dtype=jnp.float64) for y in y_list]

        def model_predict(w, X):
            return jnp.dot(X, w)

        preds_list = [model_predict(w, X_i) for X_i in X_list]

        def pairwise_loss(y_i, pred_i, y_j, pred_j):
            d = jnp.exp(-jnp.abs(y_i - y_j))
            prediction_diff = pred_i - pred_j
            return d * prediction_diff**2

        S = len(y_list)
        len_array = jnp.array([len(y) for y in y_list])
        normalization_factor = sum(
            n1 * n2
            for i, n1 in enumerate(len_array)
            for j, n2 in enumerate(len_array)
            if i != j
        )

        total_loss = 0
        for i in range(S):
            for j in range(i + 1, S):
                loss_ij = vmap(
                    lambda yi, pi: vmap(lambda yj, pj: pairwise_loss(yi, pi, yj, pj))(
                        y_list[j], preds_list[j]
                    )
                )(y_list[i], preds_list[i])
                total_loss += jnp.sum(loss_ij)

        average_loss = total_loss / normalization_factor
        return average_loss
