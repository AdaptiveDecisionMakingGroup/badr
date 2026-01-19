from badr.algorithms import SLSQP
from badr.datasets import Dataset
from badr.metrics import FairnessMetric
from badr.models import Model
from badr.oracles import ImplicitOracle, StochasticOracle


class Badr:
    def __init__(
        self,
        dset: Dataset,
        model: Model,
        metric: FairnessMetric,
        train_test: str = "train",
        oracle: str = "implicit",
        solver_cls=None,
        solver_kwargs=None,
    ) -> None:
        """
        Parameters
        ----------
        dset : Dataset
            Dataset with (X_train, y_train), (X_test, y_test), and groups.
        model : Model
            Model with set_group_weights(...), fit(...), and coef_/intercept_.
        metric : FairnessMetric
            Metric; bound to `model` if `metric.model is None`.
        train_test : {"train", "test"}, default="train"
            Which split to use.
        oracle : {"implicit", "stochastic"}, default="implicit"
            Oracle implementation to use.
        solver_cls : type, optional
            Solver class (default: SLSQP).
        solver_kwargs : dict, optional
            Keyword args passed to `solver_cls(...)`.

        Raises
        ------
        ValueError
            If `oracle` is not one of {"implicit", "stochastic"}.
        """
        if metric.model is None:
            metric.set_model(model)
        self.dset = dset
        self.model = model
        self.metric = metric
        self.train_test = train_test
        self.X = dset.X_train if train_test == "train" else dset.X_test
        self.y = dset.y_train if train_test == "train" else dset.y_test
        if oracle == "implicit":
            self.oracle = ImplicitOracle(
                dset=dset,
                model=model,
                metric=metric,
                train_test=train_test,
            )
        elif oracle == "stochastic":
            self.oracle = StochasticOracle(
                dset=dset,
                model=model,
                metric=metric,
                train_test=train_test,
            )
        else:
            raise ValueError(
                f"Unknown oracle type: {oracle}. Supported types are 'implicit' and 'stochastic'."
            )
        self.solver_cls = solver_cls or SLSQP
        self.solver_kwargs = solver_kwargs or {}
        self._solver = None

    def set_solver(self, solver):
        """
        Set a solver instance to use in `run`.

        Parameters
        ----------
        solver
            Solver instance. If `solver.oracle is None`, `run` will set it.

        Returns
        -------
        Badr
            Self.
        """
        self._solver = solver
        return self

    def run(self, **run_kwargs) -> None:
        """
        Run the solver, set group weights, refit the model, and compute outputs.

        Parameters
        ----------
        **run_kwargs
            Passed to `solver.run(**run_kwargs)`.

        Sets Attributes
        ---------------
        group_weights
            Learned group weights.
        coef_
            Fitted coefficients.
        intercept_
            Fitted intercept.
        fairness
            Metric value on the selected split.
        group_losses
            Per-group losses from the model.
        """
        if self._solver is None:  # 1) lazy‐instantiate solver if not given
            self._solver = self.solver_cls(**self.solver_kwargs)

        # 2) bind oracle if user didn’t already
        if self._solver.oracle is None:
            self._solver.set_oracle(self.oracle)
        solver = self._solver
        solver.run(**run_kwargs)
        self.group_weights = solver.group_weights
        print(f"Group weights: {self.group_weights}")
        self.model.set_group_weights(self.group_weights)
        self.model.fit(self.X, self.y, self.dset.groups)
        self.coef_ = self.model.coef_
        self.intercept_ = self.model.intercept_
        self.fairness = self.metric.fun(self.model.coef_, self.dset, self.train_test)
        self.group_losses = self.model._group_loss(self.dset)
