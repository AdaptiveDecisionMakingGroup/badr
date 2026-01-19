import jax
import jax.numpy as jnp
from typing import NamedTuple, Tuple, Union, Optional

from badr.datasets import Dataset
from badr.metrics import FairnessMetric
from badr.models import Model
from badr.oracles import Oracle


class _Batch(NamedTuple):
    X_groups: tuple
    y_groups: tuple
    counts: jnp.ndarray


class StochasticOracle(Oracle):
    """
    Oracle based on stochastic minibatches from each group.

    This oracle keeps per-group views of the data. It provides utilities to:
    - sample balanced minibatches across groups,
    - compute a stochastic estimate of the weighted lower-level objective
      ``f_hat(w, lambda) = sum_s lambda[s] * loss_s(w; batch_s)``,
    - compute gradients and Hessian-vector products w.r.t. ``w`` using JAX AD.

    Notes
    -----
    - Sampling uses a JAX PRNG key stored on the instance.
    - Per-group losses are computed via ``model._loss(w, X_batch, y_batch)``.
    - ``hvp_*`` methods use ``jax.jvp`` to avoid forming full Hessians.

    Attributes
    ----------
    X_list, y_list : list[jax.numpy.ndarray]
        Per-group feature/target arrays for the chosen split.
    S : int
        Number of groups.
    d : int
        Number of features.
    key : jax.Array
        PRNG key used for sampling.
    """

    def __init__(
        self,
        dset: Dataset,
        model: Model,
        metric: FairnessMetric,
        train_test: str = "train",
    ) -> None:
        super().__init__(dset, model, metric, train_test)
        self.X_list = [self.X[g] for g in self.groups]
        self.y_list = [self.y[g] for g in self.groups]
        self.S = len(self.X_list)
        self.n_groups = self.S
        self.d = self.X_list[0].shape[1]
        self.key = jax.random.PRNGKey(0)

    def sample_batch(
        self,
        key: jax.Array,
        batch_size: int = 32,
    ) -> Tuple[_Batch, jax.Array]:
        """
        Sample a (roughly) balanced minibatch across groups.

        The batch is split as evenly as possible: each group gets
        ``batch_size // S`` samples, and the remainder is distributed to the
        first groups.

        Parameters
        ----------
        key : jax.Array
            PRNG key.
        batch_size : int, default=32
            Total batch size across all groups.

        Returns
        -------
        _Batch
            Batch containing per-group ``X``/``y`` slices and per-group counts.
        jax.Array
            Updated PRNG key.
        """
        base = batch_size // self.S
        rem = batch_size % self.S
        want = [base + (1 if s < rem else 0) for s in range(self.S)]
        key_out = key
        Xb_groups = []
        yb_groups = []
        counts = []
        for s in range(self.S):
            Xi = self.X_list[s]
            yi = self.y_list[s]
            n_s = Xi.shape[0]
            take = min(want[s], int(n_s))
            if take > 0:
                key_out, sk = jax.random.split(key_out)
                idx = jax.random.choice(sk, n_s, shape=(take,), replace=False)
                Xb_groups.append(Xi[idx])
                yb_groups.append(yi[idx])
                counts.append(take)
            else:
                Xb_groups.append(jnp.zeros((0, self.d)))
                yb_groups.append(jnp.zeros((0,)))
                counts.append(0)
        batch = _Batch(
            X_groups=tuple(Xb_groups),
            y_groups=tuple(yb_groups),
            counts=jnp.array(counts, dtype=jnp.int32),
        )
        return batch, key_out

    def sample_group_batch(
        self,
        key: jax.Array,
        group_idx: int,
        batch_size: int = 32,
    ) -> Tuple[Tuple[jnp.ndarray, jnp.ndarray], jax.Array]:
        """
        Sample a minibatch from a single group.

        Parameters
        ----------
        key : jax.Array
            PRNG key.
        group_idx : int
            Group index in ``[0, S)``.
        batch_size : int, default=32
            Maximum number of samples to draw.

        Returns
        -------
        (jax.numpy.ndarray, jax.numpy.ndarray)
            ``(X_batch, y_batch)``.
        jax.Array
            Updated PRNG key.
        """
        Xi = self.X_list[group_idx]
        yi = self.y_list[group_idx]
        n_s = Xi.shape[0]
        m = min(batch_size, int(n_s))
        key_out = key
        if m > 0:
            key_out, sk = jax.random.split(key_out)
            idx = jax.random.choice(sk, n_s, shape=(m,), replace=False)
            Xb = Xi[idx]
            yb = yi[idx]
        else:
            Xb = jnp.zeros((0, self.d))
            yb = jnp.zeros((0,))
        return (Xb, yb), key_out

    def f_hat(self, w: jnp.ndarray, lam: jnp.ndarray, batch: _Batch) -> jnp.ndarray:
        """
        Stochastic weighted lower-level objective on a batch.

        Parameters
        ----------
        w : jax.numpy.ndarray
            Model parameters.
        lam : jax.numpy.ndarray, shape (S,)
            Group weights.
        batch : _Batch
            Per-group minibatch.

        Returns
        -------
        jax.numpy.ndarray
            Scalar value ``lam @ [loss_s(w; batch_s)]``.
        """
        vals = []
        for s in range(self.S):
            m = batch.counts[s]
            val_s = jnp.where(
                m > 0,
                self.model._loss(w, batch.X_groups[s], batch.y_groups[s]),
                0.0,
            )
            vals.append(val_s)
        vals = jnp.stack(vals)
        return jnp.dot(lam, vals)

    def grad_w_f_hat(
        self, w: jnp.ndarray, lam: jnp.ndarray, batch: _Batch
    ) -> jnp.ndarray:
        """
        Gradient of :meth:`f_hat` w.r.t. ``w``.

        Parameters
        ----------
        w : jax.numpy.ndarray
            Model parameters.
        lam : jax.numpy.ndarray, shape (S,)
            Group weights.
        batch : _Batch
            Per-group minibatch.

        Returns
        -------
        jax.numpy.ndarray
            Gradient vector.
        """
        return jax.grad(lambda ww: self.f_hat(ww, lam, batch))(w)

    def hvp_w_f_hat(
        self, w: jnp.ndarray, lam: jnp.ndarray, v: jnp.ndarray, batch: _Batch
    ) -> jnp.ndarray:
        """
        Hessian-vector product of :meth:`f_hat` w.r.t. ``w``.

        Parameters
        ----------
        w : jax.numpy.ndarray
            Model parameters.
        lam : jax.numpy.ndarray, shape (S,)
            Group weights.
        v : jax.numpy.ndarray
            Vector to multiply by the Hessian.
        batch : _Batch
            Per-group minibatch.

        Returns
        -------
        jax.numpy.ndarray
            ``H_w(f_hat) @ v``.
        """
        gfun = jax.grad(lambda ww: self.f_hat(ww, lam, batch))
        _, hvp = jax.jvp(gfun, (w,), (v,))
        return hvp

    def grad_lower_groups(
        self,
        w: jnp.ndarray,
        batch_or_size: Union[_Batch, int],
        key: Optional[jax.Array] = None,
    ) -> jnp.ndarray:
        """
        Per-group gradients of the group losses at ``w``.

        Parameters
        ----------
        w : jax.numpy.ndarray
            Model parameters.
        batch_or_size : _Batch or int
            Either an explicit batch, or a batch size to sample.
        key : jax.Array, optional
            PRNG key used when sampling.

        Returns
        -------
        jax.numpy.ndarray, shape (S, n_params)
            Stack of gradients, one per group.
        """
        if isinstance(batch_or_size, _Batch):
            batch = batch_or_size
        else:
            size = int(batch_or_size)
            if key is None:
                batch, self.key = self.sample_batch(self.key, size)
            else:
                batch, _ = self.sample_batch(key, size)
        grads = []
        for s in range(self.S):
            m = batch.counts[s]

            def F_s(ww):
                return jnp.where(
                    m > 0,
                    self.model._loss(ww, batch.X_groups[s], batch.y_groups[s]),
                    0.0,
                )

            g_s = jax.grad(F_s)(w)
            grads.append(g_s)
        return jnp.stack(grads)

    def hvp_lower_group(
        self,
        w: jnp.ndarray,
        v: jnp.ndarray,
        group_idx: int,
        batch_size: int = 32,
        key: Optional[jax.Array] = None,
    ) -> jnp.ndarray:
        """
        Hessian-vector product for a single group's loss.

        Parameters
        ----------
        w : jax.numpy.ndarray
            Model parameters.
        v : jax.numpy.ndarray
            Vector to multiply by the Hessian.
        group_idx : int
            Group index.
        batch_size : int, default=32
            Batch size for the group minibatch.
        key : jax.Array, optional
            PRNG key used when sampling.

        Returns
        -------
        jax.numpy.ndarray
            ``H_w(loss_group) @ v``.
        """
        if key is None:
            (Xb, yb), self.key = self.sample_group_batch(
                self.key, group_idx, batch_size
            )
        else:
            (Xb, yb), _ = self.sample_group_batch(key, group_idx, batch_size)
        m = Xb.shape[0]

        def F_s(ww):
            return jnp.where(m > 0, self.model._loss(ww, Xb, yb), 0.0)

        return jax.jvp(jax.grad(F_s), (w,), (v,))[1]

    def jt_v_lambda_of_grad_w_f_hat(
        self, w: jnp.ndarray, v: jnp.ndarray, batch: _Batch
    ) -> jnp.ndarray:
        """
        Compute J^T v where J = d/dlambda (grad_w f_hat).

        For ``f_hat(w, lambda) = sum_s lambda[s] * loss_s(w)``,
        ``grad_w f_hat = sum_s lambda[s] * grad_w loss_s`` and the Jacobian
        w.r.t. ``lambda`` has columns ``grad_w loss_s``. This method returns the
        vector of dot-products ``[<grad_w loss_s, v>]`` over groups.

        Parameters
        ----------
        w : jax.numpy.ndarray
            Model parameters.
        v : jax.numpy.ndarray
            Vector in parameter space.
        batch : _Batch
            Per-group minibatch.

        Returns
        -------
        jax.numpy.ndarray, shape (S,)
            ``[dot(grad_w loss_s, v)]_s``.
        """
        dots = []
        for s in range(self.S):
            m = batch.counts[s]

            def F_s(ww):
                return jnp.where(
                    m > 0,
                    self.model._loss(ww, batch.X_groups[s], batch.y_groups[s]),
                    0.0,
                )

            g_s = jax.grad(F_s)(w)
            dots.append(jnp.dot(g_s, v))
        return jnp.stack(dots)

    def grad_upper(self, w: jnp.ndarray, lam: jnp.ndarray):
        """
        Upper-level gradients at ``w``.

        Computes the gradient of the fairness metric w.r.t. ``w``. The gradient
        w.r.t. ``lam`` is zero here because the metric is evaluated only on
        ``w`` (no direct dependence on ``lam`` in this implementation).

        Parameters
        ----------
        w : jax.numpy.ndarray
            Model parameters.
        lam : jax.numpy.ndarray
            Group weights.

        Returns
        -------
        (jax.numpy.ndarray, jax.numpy.ndarray)
            ``(d metric / d w, d metric / d lam)``.
        """
        grad_H_w = jax.grad(lambda ww: self.metric.fun(ww, self.dset, self.train_test))(
            w
        )
        grad_H_lambda = jnp.zeros_like(lam)
        return grad_H_w, grad_H_lambda
