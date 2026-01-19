import jax.numpy as jnp
from jax import grad, hessian

from badr.datasets import Dataset
from badr.metrics import FairnessMetric
from badr.models import Model
from badr.oracles import Oracle


class ImplicitOracle(Oracle):
    """
    Oracle using implicit differentiation through the lower-level optimum.

    For given group weights ``lambda``, the lower-level model is fit with these
    weights. The gradient uses the implicit function theorem: it forms a mixed
    Hessian of the lower-level loss (weighted sum over groups) and solves linear
    systems to obtain ``dw*/dlambda``. The final gradient is
    ``(dw*/dlambda)^T @ d(metric)/dw`` evaluated at ``w*``.

    Notes
    -----
    - Requires the model to provide a differentiable scalar loss ``model._loss(w, X, y)``.
    - Uses JAX ``grad`` and ``hessian``; the solve uses ``jax.numpy.linalg.solve``.

    Methods
    -------
    solve_lower(group_weights)
        Fit the model with ``group_weights`` and return the resulting coefficients.
    w_star_grad(group_weights)
        Compute ``dw*/dgroup_weights`` at the current solution.
    """

    def __init__(
        self,
        dset: Dataset,
        model: Model,
        metric: FairnessMetric,
        train_test: str = "train",
    ) -> None:
        super().__init__(dset, model, metric, train_test)

    def solve_lower(self, group_weights: jnp.ndarray) -> jnp.ndarray:
        """
        Solve the lower-level problem for fixed group weights.

        Parameters
        ----------
        group_weights : jax.numpy.ndarray, shape (n_groups,)
            Group weights used by the model during training.

        Returns
        -------
        jax.numpy.ndarray
            Coefficient vector at the fitted solution.
        """
        self.model.set_group_weights(group_weights)
        self.model.fit(self.X, self.y, self.groups)
        self.coef_ = self.model.coef_
        self.intercept_ = self.model.intercept_
        return self.coef_

    def w_star_grad(self, group_weights: jnp.ndarray) -> jnp.ndarray:
        """
        Jacobian of the lower-level optimum ``w*`` w.r.t. group weights.

        Parameters
        ----------
        group_weights : jax.numpy.ndarray, shape (n_groups,)
            Group weights.

        Returns
        -------
        jax.numpy.ndarray, shape (n_params, n_groups)
            Matrix whose column ``i`` is ``dw*/dlambda_i``.
        """
        self.X_list = [self.X[g] for g in self.groups]
        self.y_list = [self.y[g] for g in self.groups]

        grad_LL, hessian_LL = grad(self.model._loss), hessian(self.model._loss)
        mixed_hessian = sum(
            group_weights[i] * hessian_LL(self.coef_, self.X_list[i], self.y_list[i])
            for i in range(self.n_groups)
        )
        res = jnp.zeros((self.X_list[0].shape[1], len(self.X_list)))
        for i in range(len(self.X_list)):
            v_i = -jnp.linalg.solve(  # why not CVXPy here ?
                mixed_hessian, grad_LL(self.coef_, self.X_list[i], self.y_list[i])
            )
            res = res.at[:, i].set(v_i)
        return res

    def fun(self, group_weights):
        """
        Evaluate the upper-level metric at the fitted lower-level solution.

        Parameters
        ----------
        group_weights : jax.numpy.ndarray, shape (n_groups,)
            Group weights.

        Returns
        -------
        ArrayLike
            Metric value ``metric.fun(w*, dset, train_test)``.
        """
        w_star = self.solve_lower(group_weights)
        return self.metric.fun(w_star, self.dset, self.train_test)

    def grad(self, group_weights):
        """
        Gradient of the upper-level metric w.r.t. group weights.

        Parameters
        ----------
        group_weights : jax.numpy.ndarray, shape (n_groups,)
            Group weights.

        Returns
        -------
        ArrayLike
            Gradient vector of shape ``(n_groups,)``.
        """
        w_star = self.solve_lower(group_weights)
        upper_grad = self.metric.grad(w_star, self.dset, self.train_test)
        return self.w_star_grad(group_weights).T @ upper_grad
