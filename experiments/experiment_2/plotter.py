def plot_results(
    results_path: str = "results.jsonl",
    alpha: float = 0.8,
    dpi: int = 150,
):
    """ECDF view of fairness metrics across methods."""
    import textwrap

    def set_wrapped_ylabel(ax, text, width=10, **label_kwargs):
        wrapped = "\n".join(textwrap.wrap(text, width=width))
        ax.set_ylabel(wrapped, rotation=90, va="center", **label_kwargs)

    from collections import defaultdict
    import json

    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.lines import Line2D
    from tueplots import axes as tp_axes
    from tueplots import figsizes

    def _load_results(path):
        grouped = defaultdict(list)
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                algo = rec.get("algo")
                if algo is not None:
                    grouped[algo].append(rec)
        return grouped

    def _expand_range(rng, frac=0.03):
        """Add relative padding to a (min, max) tuple."""
        if rng is None:
            return None
        lo, hi = rng
        if not np.isfinite(lo) or not np.isfinite(hi):
            return rng
        if hi == lo:
            pad = abs(lo) * frac if lo != 0 else frac
            return lo - pad, hi + pad
        pad = (hi - lo) * frac
        return lo - pad, hi + pad

    res = _load_results(results_path)
    if not res:
        print("No results found.")
        return

    # method_keys = list(res.keys())
    # print(f"Methods found: {method_keys}")
    method_keys = ["badr", "erm", "one_fit", "balanced", "minmax"]
    method_colors = {
        "badr": "#C81919",
        "erm": "#662C91",
        "balanced": "#DA9F93",
        "one_fit": "#A0AF63",
        "minmax": "#6699CC",
    }
    method_linestyles = {
        "badr": "solid",
        "erm": "dashed",
        "balanced": "dashdot",
        "one_fit": "solid",
        "minmax": ":",
    }
    method_labels = {
        "badr": "badr",
        "erm": "Uniform sampling",
        "balanced": "Balanced sampling",
        "one_fit": "One-group fitting",
        "minmax": "Minimax fairness",
    }
    method_zorder = {m: (10 if m == "badr" else 1) for m in method_keys}

    metric_names = sorted(
        {
            rec.get("metric")
            for lst in res.values()
            for rec in lst
            if rec.get("metric") is not None
        }
    )

    def _select(method, metric_name):
        return [e for e in res[method] if e.get("metric") == metric_name]

    def _plot_ecdf(ax, values, color, z, linestyle="-"):
        if not values.size:
            return False
        x = np.sort(values)
        y = np.arange(1, x.size + 1) / x.size
        ax.step(
            x,
            y,
            where="post",
            color=color,
            linewidth=1.5,
            alpha=alpha,
            linestyle=linestyle,
            zorder=z,
        )
        return True

    if not metric_names:
        print("No metrics found.")
        return

    with plt.rc_context(figsizes.jmlr2001(nrows=max(1, len(metric_names)), ncols=3)):
        plt.rcParams.update({"figure.dpi": dpi})
        plt.rcParams.update(tp_axes.tick_direction(x="out", y="out"))
        plt.rcParams["font.family"] = "Open Sans"
        plt.rcParams["font.weight"] = "light"
        plt.rcParams["font.size"] = 8.95
        plt.rcParams["axes.facecolor"] = "white"

        fig, axes = plt.subplots(
            nrows=len(metric_names), ncols=2, constrained_layout=True
        )
        axes = [axes] if len(metric_names) == 1 else list(axes)

        legend_handles, legend_labels = [], []

        for i, metric_name in enumerate(metric_names):
            left_ax, right_ax = axes[i]
            train_all, test_all, vals = [], [], {}

            for method in method_keys:
                label = method_labels.get(method, method)
                color = method_colors.get(
                    method, plt.cm.tab10(method_keys.index(method) % 10)
                )
                entries = _select(method, metric_name)
                train_vals = [e["train_metric"] for e in entries if "train_metric" in e]
                test_vals = [e["test_metric"] for e in entries if "test_metric" in e]
                train_arr = np.asarray(train_vals, dtype=float)
                test_arr = np.asarray(test_vals, dtype=float)
                vals[method] = (train_arr, test_arr, label, color)
                train_all.extend(train_arr.tolist())
                test_all.extend(test_arr.tolist())

            if not train_all and not test_all:
                left_ax.set_visible(False)
                right_ax.set_visible(False)
                continue

            train_range = _expand_range(
                (min(train_all), max(train_all)) if train_all else None, frac=0.03
            )
            test_range = _expand_range(
                (min(test_all), max(test_all)) if test_all else None, frac=0.03
            )

            for method in method_keys:
                train_vals, test_vals, label, color = vals[method]
                z = method_zorder.get(method, 1)
                ls = method_linestyles.get(method, "solid")

                if i == 0 and (train_vals.size or test_vals.size):
                    legend_handles.append(
                        Line2D(
                            [0],
                            [0],
                            color=color,
                            linewidth=1.5,
                            alpha=alpha,
                            linestyle=ls,
                        )
                    )
                    legend_labels.append(label)

                _plot_ecdf(left_ax, train_vals, color, z, ls)
                _plot_ecdf(right_ax, test_vals, color, z, ls)

            set_wrapped_ylabel(left_ax, metric_name, width=12, labelpad=10)

            for ax, xrng in ((left_ax, train_range), (right_ax, test_range)):
                if xrng is not None:
                    ax.set_xlim(xrng)
                ax.set_ylim(0, 1)
                ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
                ax.grid(
                    True,
                    which="both",
                    axis="both",
                    linestyle=":",
                    linewidth=0.5,
                    alpha=0.6,
                )

        axes[-1][0].set_xlabel("Fairness on train")
        axes[-1][1].set_xlabel("Fairness on test")

        if legend_handles:
            fig.legend(
                legend_handles,
                legend_labels,
                loc="lower center",
                ncol=3,
                bbox_to_anchor=(0.5, -0.07),
            )
        plt.savefig("../../figures/experiment_2.pdf", bbox_inches="tight")
        # plt.show()


if __name__ == "__main__":
    plot_results(
        results_path="results.jsonl",
        alpha=1,
    )
