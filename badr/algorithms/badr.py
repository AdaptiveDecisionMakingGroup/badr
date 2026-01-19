import time
import jax.numpy as jnp
from .__base__ import Algorithm


class BADRSGD(Algorithm):
    """
    Stochastic BADR updates over (w, v, lambda) with simplex projection.

    Uses a :class:`~badr.oracles.StochasticOracle`-style interface:
    draws a minibatch, forms a group-weighted inner gradient for ``w``, updates an
    auxiliary vector ``v`` using Hessian-vector products, and updates ``lambda``
    via a clipped step followed by projection onto the simplex.

    Parameters
    ----------
    w0 : jax.numpy.ndarray
        Initial parameter vector for the lower-level variable ``w``.
    batch_size : int, default=1
        Total minibatch size across groups.
    step_w : float, default=1e-1
        Step size for ``w`` and ``v`` updates (used as ``step_1``).
    step_v : float, default=1e-1
        Stored step size for ``v`` (currently not used in the shown code).
    step_lambda : float, default=1.0
        Step size for ``lambda`` update (used as ``step_2``).
    clip_value : float, default=1.0
        L2-norm clipping threshold applied to the ``lambda`` gradient estimate.

    Attributes
    ----------
    primal_solution : jax.numpy.ndarray or None
        Final ``w`` iterate.
    aux_solution : jax.numpy.ndarray or None
        Final ``v`` iterate.
    message : str or None
        Status message after :meth:`run`.
    history_w, history_v, history_lambda : list[jax.numpy.ndarray]
        Iterates recorded during :meth:`run`.
    history_inner_loss_batch : list[float]
        ``f_hat(w, lambda; batch)`` over iterations.
    history_outer_metric : list[float]
        Metric value on the full split at current ``w``.
    history_norm_grad_w, history_norm_v, history_norm_jt_v, history_norm_grad_H_w : list[float]
        Diagnostic norms recorded per iteration.
    history_lambda_entropy : list[float]
        Entropy ``-sum(lambda log lambda)`` per iteration.
    history_clip_fraction : list[float]
        Running fraction of iterations where lambda-gradient clipping activated.

    Notes
    -----
    - Simplex projection uses sorting-based Euclidean projection.
    - Clipping is detected by comparing the pre/post L2 norms of the gradient.
    """

    def __init__(
        self,
        w0: jnp.ndarray,
        batch_size: int = 1,
        step_w: float = 1e-1,
        step_v: float = 1e-1,
        step_lambda: float = 1.0,
        clip_value: float = 1.0,
    ) -> None:
        super().__init__("BADRSGD")
        self.w0 = jnp.array(w0)
        self.batch_size = batch_size
        self.step_w = step_w
        self.step_v = step_v
        self.step_lambda = step_lambda
        self.clip_value = clip_value

        # existing histories
        self.history_w = []
        self.history_v = []
        self.history_lambda = []
        self.history_time = []
        self.message = None
        self.primal_solution = None
        self.aux_solution = None

        # NEW: metric histories
        self.history_inner_loss_batch = []
        self.history_outer_metric = []
        self.history_norm_grad_w = []
        self.history_norm_v = []
        self.history_norm_jt_v = []
        self.history_norm_grad_H_w = []
        self.history_lambda_entropy = []
        self.history_clip_fraction = []  # running fraction in [0,1]

    @staticmethod
    def _project_simplex(v: jnp.ndarray, radius: float = 1.0) -> jnp.ndarray:
        v = jnp.asarray(v)
        v_sorted = jnp.sort(v)[::-1]
        cssv = jnp.cumsum(v_sorted) - radius
        ind = jnp.arange(1, v.shape[0] + 1)
        cond = v_sorted - cssv / ind > 0
        rho = max(int(jnp.sum(cond)) - 1, 0)
        theta = cssv[rho] / (rho + 1)
        return jnp.maximum(v - theta, 0.0)

    def _clip(self, g: jnp.ndarray) -> jnp.ndarray:
        norm = jnp.linalg.norm(g)
        factor = jnp.minimum(1.0, self.clip_value / (norm + 1e-12))
        return factor * g

    def run(self, max_iter: int = 1000, verbose: int = 1, trace: bool = False):
        """
        Run stochastic BADR iterations.

        Parameters
        ----------
        max_iter : int, default=1000
            Number of iterations.
        verbose : int, default=1
            If > 0, prints a completion message.
        trace : bool, default=False
            If True, also appends the outer metric to ``history_f`` each iteration.

        Returns
        -------
        jax.numpy.ndarray
            Final group weights ``lambda``.

        Raises
        ------
        ValueError
            If no oracle has been set.
        """
        if self.oracle is None:
            raise ValueError("Oracle not set. Please set the oracle before running.")

        w = jnp.array(self.w0)
        v = jnp.zeros_like(w)
        lmbda = jnp.ones(self.oracle.n_groups) / self.oracle.n_groups

        # reset histories
        self.history_w = [w]
        self.history_v = [v]
        self.history_lambda = [lmbda]
        self.history_time = [0.0]
        if trace:
            self.history_f = [
                float(
                    self.oracle.metric.fun(w, self.oracle.dset, self.oracle.train_test)
                )
            ]
        else:
            self.history_f = []

        # also reset metric histories
        self.history_inner_loss_batch = []
        self.history_outer_metric = []
        self.history_norm_grad_w = []
        self.history_norm_v = []
        self.history_norm_jt_v = []
        self.history_norm_grad_H_w = []
        self.history_lambda_entropy = []
        self.history_clip_fraction = []

        start_t = time.perf_counter()
        step_1 = self.step_w
        step_2 = self.step_lambda
        clipped_count = 0  # running count of clipped λ-updates

        for t in range(max_iter):
            # --- draw ONE inner batch and reuse it for all inner terms (SOBA)
            batch, self.oracle.key = self.oracle.sample_batch(
                self.oracle.key, self.batch_size
            )

            # 1) inner gradient g_w = ∇_w f_inner(w, λ; batch)
            grad_F = self.oracle.grad_lower_groups(w, batch)  # shape (S, d)
            grad_w = grad_F.T @ lmbda  # shape (d,)
            w_next = w - step_1 * grad_w

            # 2) outer gradient pieces (metric)
            grad_H_w, grad_H_lambda = self.oracle.grad_upper(w, lmbda)

            # 3) v-update: v ← v - ρ( H_{ww} v + ∇_w H )
            hvp_weighted = self.oracle.hvp_w_f_hat(w, lmbda, v, batch)  # shape (d,)
            v_next = v - step_1 * (grad_H_w + hvp_weighted)

            # 4) λ-update: λ ← Proj_Δ( λ - γ * Clip( J^T v + ∇_λ H ) )
            jt_v = self.oracle.jt_v_lambda_of_grad_w_f_hat(w, v, batch)  # shape (S,)
            lambda_grad_est = jt_v + grad_H_lambda
            # detect clipping (norm reduced)
            pre_norm = float(jnp.linalg.norm(lambda_grad_est))
            clipped = self._clip(lambda_grad_est)
            post_norm = float(jnp.linalg.norm(clipped))
            did_clip = 1.0 if post_norm + 1e-12 < pre_norm else 0.0
            clipped_count += did_clip

            lambda_candidate = lmbda - step_2 * clipped
            lambda_next = self._project_simplex(lambda_candidate)

            # step
            w, v, lmbda = w_next, v_next, lambda_next

            # --- logging ---
            # inner loss on the sampled batch
            inner_loss_batch = float(self.oracle.f_hat(w, lmbda, batch))
            # outer metric (current w on full data; same as your trace metric)
            outer_metric_val = float(
                self.oracle.metric.fun(w, self.oracle.dset, self.oracle.train_test)
            )

            # norms
            norm_grad_w = float(jnp.linalg.norm(grad_w))
            norm_v = float(jnp.linalg.norm(v))
            norm_jt_v = float(jnp.linalg.norm(jt_v))
            norm_grad_H_w = float(jnp.linalg.norm(grad_H_w))

            # λ entropy
            lambda_entropy = float(-jnp.sum(lmbda * jnp.log(lmbda + 1e-12)))

            # fraction of clipped λ-updates up to now
            clip_fraction = clipped_count / float(t + 1)

            # bookkeeping
            elapsed = time.perf_counter() - start_t
            self.history_w.append(w)
            self.history_v.append(v)
            self.history_lambda.append(lmbda)
            self.history_time.append(elapsed)
            if trace:
                self.history_f.append(outer_metric_val)

            # store metrics
            self.history_inner_loss_batch.append(inner_loss_batch)
            self.history_outer_metric.append(outer_metric_val)
            self.history_norm_grad_w.append(norm_grad_w)
            self.history_norm_v.append(norm_v)
            self.history_norm_jt_v.append(norm_jt_v)
            self.history_norm_grad_H_w.append(norm_grad_H_w)
            self.history_lambda_entropy.append(lambda_entropy)
            self.history_clip_fraction.append(clip_fraction)

        self.primal_solution = w
        self.aux_solution = v
        self.group_weights = lmbda
        self.message = f"Completed {max_iter} (stochastic) iterations."
        if verbose > 0:
            print("[BADR] Message:", self.message)
        return self.group_weights
