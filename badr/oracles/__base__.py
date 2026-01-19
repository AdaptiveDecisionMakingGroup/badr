import jax.numpy as jnp
from jax.typing import ArrayLike

from badr.datasets import Dataset
from badr.metrics import FairnessMetric
from badr.models import Model


class Oracle:
    """
    Base oracle for the bilevel BADR objective.

    An oracle takes group weights ``lambda`` and returns the upper-level objective
    value and gradient w.r.t. ``lambda``. Subclasses differ in how they handle the
    lower-level optimization over model parameters.

    Parameters
    ----------
    dset : Dataset
        Dataset providing splits and group indices.
    model : Model
        Model used for the lower-level problem.
    metric : FairnessMetric
        Upper-level metric evaluated at the lower-level solution.
    train_test : {"train", "test"}, default="train"
        Split used to form groups and evaluate the metric.

    Attributes
    ----------
    X, y
        Feature matrix and targets for the chosen split.
    groups : list[np.ndarray]
        Group indices for the chosen split.
    n_groups : int
        Number of groups.
    """

    def __init__(
        self,
        dset: Dataset,
        model: Model,
        metric: FairnessMetric,
        train_test: str = "train",
    ) -> None:
        self.dset = dset
        if train_test == "train":
            self.X = dset.X_train
            self.y = dset.y_train
            self.groups = dset.groups
        elif train_test == "test":
            self.X = dset.X_test
            self.y = dset.y_test
            self.groups = dset.groups_test
        else:
            raise ValueError("train_test should be either 'train' or 'test'.")
        self.n_groups = dset.n_groups
        self.model = model
        self.metric = metric
        self.train_test = train_test

    def fun(self, group_weights: jnp.ndarray) -> ArrayLike:
        """
        Upper-level objective value at ``group_weights``.

        Parameters
        ----------
        group_weights : jax.numpy.ndarray, shape (n_groups,)
            Group weights (often denoted ``lambda``).

        Returns
        -------
        ArrayLike
            Scalar objective value.

        Raises
        ------
        NotImplementedError
            If not implemented.
        """
        raise NotImplementedError("This method should be implemented in subclasses.")

    def grad(self, group_weights: jnp.ndarray) -> ArrayLike:
        """
        Gradient of :meth:`fun` w.r.t. ``group_weights``.

        Parameters
        ----------
        group_weights : jax.numpy.ndarray, shape (n_groups,)
            Group weights.

        Returns
        -------
        ArrayLike
            Gradient vector of shape ``(n_groups,)``.

        Raises
        ------
        NotImplementedError
            If not implemented.
        """
        raise NotImplementedError("This method should be implemented in subclasses.")
