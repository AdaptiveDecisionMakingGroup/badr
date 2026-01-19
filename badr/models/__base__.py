from typing import List

import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike


class Model:
    """
    Base model interface.
    """

    def __init__(self) -> None:
        self.coef_: jnp.ndarray = jnp.array([])
        self.intercept_: float = 0.0
        self.l2_reg: float = 0.0

    def set_group_weights(self, group_weights: jnp.ndarray) -> None:
        """
        Set per-group weights.

        Parameters
        ----------
        group_weights : jax.numpy.ndarray
            Array of shape ``(n_groups,)``.
        """
        self.group_weights = group_weights

    def _group_loss(
        self, dset, train_test: str = "train", w: jnp.ndarray = jnp.array([])
    ):
        """
        Per-group loss values.

        Parameters
        ----------
        dset
            Dataset object.
        train_test : {"train", "test"}, default="train"
            Split to use.
        w : jax.numpy.ndarray, default=jnp.array([])
            Optional parameter vector. If empty, use fitted parameters.

        Returns
        -------
        ArrayLike
            Per-group losses.

        Raises
        ------
        NotImplementedError
            If not implemented by the subclass.
        """
        raise NotImplementedError(
            "Model._group_loss() must be implemented in subclasses."
        )

    def _loss(self, w: jnp.ndarray, X: np.ndarray, y: np.ndarray) -> ArrayLike:
        """
        Scalar loss on a single (X, y) split.

        Parameters
        ----------
        w : jax.numpy.ndarray
            Parameter vector.
        X : numpy.ndarray
            Feature matrix.
        y : numpy.ndarray
            Targets.

        Returns
        -------
        ArrayLike
            Scalar loss value.

        Raises
        ------
        NotImplementedError
            If not implemented by the subclass.
        """
        raise NotImplementedError("This method should be implemented in subclasses.")

    def fit(self, X: np.ndarray, y: np.ndarray, groups: List[np.ndarray]):
        """
        Fit on grouped data.

        Parameters
        ----------
        X : numpy.ndarray
            Feature matrix.
        y : numpy.ndarray
            Targets.
        groups : list[numpy.ndarray]
            Per-group selectors (integer indices or boolean masks).

        Returns
        -------
        Model
            Self.

        Raises
        ------
        NotImplementedError
            If not implemented by the subclass.
        """
        raise NotImplementedError("This method should be implemented in subclasses.")
