# Getting started
<style>
.md-content .md-typeset h1 { display: none; }
</style>
<p align="center">
    <span style="font-size:2rem; font-weight:bold; display:block; text-align:left;">
      Getting started
    </span>
</p>


This quick start tutorial will get you going with ```BADR```.

## Installation

The recommended way to install **BADR** is with [uv](https://docs.astral.sh/uv/).
```bash
$ uv init
$ uv add badr
```
**BADR** is also available on PyPi, hence it can be installed very simply by running the following command line:
```bash
$ pip install -U badr
```

!!! warning "Environments"

    Generally, you do not want to modify Python's installation environment. Rather, it is a best practice to do everything within isolated virtual environments. This is not something ```pip``` enforces by default.

Now that you're all set, you should be able to run

```python
import badr as bdr
```

## Instantiation of your fairness problem
To define your fairness problem, you need three core elements:

### A Dataset object.
You can fetch one of the available datasets with
```python
dset = bdr.datasets.fetch_adult(
    n_groups = 3,
    test_size = 0.2
)
```
or transform your own ```pandas``` DataFrame in a Dataset object with
```python
from badr.datasets import load_dataframe
df = pd.read_csv(...) # Load your dataframe.
dset = load_dataframe(
    df = df,
    target_col = "y",
    sensitive_cols = ["sex"]
)
```
### A FairnessMetric object.
Fairness metrics are available through the **FairnessMetric** class. You can choose one as simply as
```python
metric = bdr.metrics.IndividualFairness()
```
### A Model object.
Learning models in ```BADR``` inherit from ```scikit-learn``` estimators and classifier classes. For example, you can define a Linear Regression model in one line:
```python
model = bdr.models.RidgeRegression(
    l2_reg = 1e-1
)
```
!!! warning "Setting the model"

    If you do not wish to use the **Badr** object afterwards, it is also important to link your metric to your model with
    ```python
    metric.set_model(model)
    ```
    This is useful for metrics such as the *Group Variance* metric which measures fairness using group losses rather than using predictions.

##  Learning a fair estimator
Once you have defined your three core elements, you can build a Pareto-fair estimator.

For that, you need to wrap your dataset, model and fairness metric inside an ```Badr``` object and run it.

```python
badr = bdr.Badr(dset, model, metric)
badr.run()
```

You are already done ! You can get back your fair estimator with ```badr.model``` or retrieve values of interest with
```python
badr_test_score = badr.model.score(dset.X_test, dset.y_test)
badr_fairness = metric.fun(badr.coef_, dset)
```

You can find more information on the ```Badr``` object in the [API Reference](../reference/index.md).