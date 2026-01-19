<style>
.md-content .md-typeset h1 { display: none; }
.md-content .md-typeset ul li a {
    font-weight: 600;
    color: var(--md-accent-fg-color);
  }
</style>

<p align="center">
  <span style="font-size:2rem; font-weight:bold; display:block; text-align:center;">
    &lt; BADR &gt;
  </span>
</p>

<p align="center" style="font-size:1rem">
  <em>A bilevel adaptive rescalarization framework to recover Pareto-efficient fair models.</em>
</p>


Welcome to the documentation for **BADR**, an open-source Python toolbox for training Pareto-efficient fair machine learning models.

Many fair ML methods can produce *Pareto-inefficient* solutions: the performance of some groups could be improved without hurting others. This happens frequently with traditional in-processing techniques such as fairness-through-regularization. BADR is designed to directly address this issue.


## What is BADR?

BADR implements a bilevel adaptive rescalarization framework that can recover the optimal Pareto-efficient model for a chosen fairness metric.

At a high level:

- **Lower level:** solves a weighted empirical risk minimization (ERM) problem, where weights are a convex combination of groups.
- **Upper level:** optimizes the selected fairness objective, adapting the group weights to target the desired notion of fairness.

BADR is *metric-agnostic*: it adapts to a broad range of fairness metrics studied in the literature, rather than committing to a single “perspective” on fairness.

The toolbox provides two scalable, single-loop algorithms (BADR-GD and BADR-SGD) and integrates seamlessly with **scikit-learn** models to make experimentation on real datasets straightforward.


### A quick example

Train a Pareto-efficient model for a chosen fairness metric, then evaluate predictive performance and the fairness objective:

```python
import badr as bdr

# --- Problem setup ---
dset = bdr.datasets.fetch_adult()
model = bdr.models.LogisticRegression()
metric = bdr.metrics.IndividualFairness()

# --- Run BADR ---
badr = bdr.Badr(dset, model, metric)
badr.run()

# --- Evaluate ---
test_score = badr.model.score(dset.X_test, dset.y_test)
fairness_value = metric.fun(badr.coef_, dset)
```

## How this documentation is organized

- The [Getting Started](getting-started/index.md) section will help you install BADR and build your first Pareto-fair model.
- The [API Reference](reference/index.md) contains a complete reference for BADR classes, functions, and metrics. This is useful when you want exact signatures and options.