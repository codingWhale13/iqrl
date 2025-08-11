import wandb
import pandas as pd
import pylab as plt
import matplotlib.ticker as mticker
import numpy as np
import seaborn as sns
import rliable.library as rly
import rliable.metrics as metrics
import numpy as np
import os


# === Constants ===
ENTITY_AND_PROJECT = "kielen1-aalto-university/iqrl"
EXPERIMENT = "exp_paper"
OUT_DIR = "results"
os.makedirs(os.path.join(OUT_DIR, EXPERIMENT), exist_ok=True)

BENCHMARKS = ["walker-all", "spin-dog", "final-4"]
TASKS = {
    BENCHMARKS[0]: ["walker-stand", "walker-walk", "walker-run"],
    BENCHMARKS[1]: ["finger-spin", "dog-walk"],
    BENCHMARKS[2]: [
        "cartpole-balance",
        "cup-catch",
        "finger-spin",
        "pendulum-swingup",
    ],
}

EPISODE_COUNT = 1000
EPISODE_LEN = 1000
RLIABLE_REPS = 1000  # TODO: Use 50000, as for final performance plots

EVAL = "eval/"
ENV_STEP = "env_step"
EPISODIC_RETURN = "episodic_return"
EPISODIC_RETURN_MEAN = "episodic_return_mean"


# Colors from https://sashamaps.net/docs/resources/20-colors/
COLORS15_RGB = {
    # MT-TD3-Latent & baselines
    "MT-TD3-Direct": (0, 130, 200),  # Blue
    "MT-TD3-Latent": (230, 25, 75),  # Red
    "MT-TD3-Latent 1-hot": (230, 25, 75),  # Red
    "MT-TD3-Latent Label": (230, 25, 75),  # Red
    "MT-TD3-Latent FiLM": (230, 25, 75),  # Red
    "MT-TD3-Latent Concat": (230, 25, 75),  # Red
}
COLORS = {k: (r / 255, g / 255, b / 255) for k, (r, g, b) in COLORS15_RGB.items()}
FONT_SIZE = 16  # Labels, legend, markers
FONT_SIZE_TITLE = 20

# === Specify W&B run IDs===
algos = ["MT-TD3-Latent", "MT-TD3-Direct"]
RUN_IDS = {
    BENCHMARKS[0]: {
        algos[0]: ["1zp9zqip", "ra7b5sr3", "aenmg9ig", "7150h1oc", "v5wkvaut"],
        algos[1]: ["ef5evw0d", "llpqc46y", "n7owite3", "j2y6k2hk", "wi7fyi1o"],
    },
    BENCHMARKS[1]: {
        algos[0]: ["me4qpu7w", "lqd17o0w", "nbwt66h6", "h3n98fmj", "shuxbe3r"],
        algos[1]: ["2y76stwi", "bis3oe6h", "dpgyng52", "hfksy4km", "nuk8x7nh"],
    },
    BENCHMARKS[2]: {
        algos[0]: ["qduq3258", "0ohltkcw", "15t6dphm", "78vm2o3k", "yqk11i6w"],
        algos[1]: ["omgie5ti", "28gu5eye", "5ib9yw44", "jyjx1vdw", "q21pje76"],
    },
}


# HELPERS
def aggregate_func(x):
    return np.array([metrics.aggregate_iqm(x)])


def millions(x, pos):
    # Format large x-axis numbers (like 1e6) as '1M'
    if int(x) == 0:
        return "0"
    elif x < 1e6:
        return f"{str(int(x))[:3]}k"
    else:
        return f"{x * 1e-6:.0f}M"


max_n_tasks = max(len(TASKS[benchmark]) for benchmark in BENCHMARKS)
for benchmark in BENCHMARKS:
    tasks = TASKS[benchmark]
    n_tasks = len(tasks)

    # Get data from W&B and compute confidence intervals
    wandb_api = wandb.Api()
    task_dfs = []
    for task in tasks:
        print(f"WORKING ON {benchmark} {task}")
        metric_key = f"eval/.{task}.episodic_return"
        x_axis = f"eval/.{task}.env_step"

        # Elements will be DataFrame with keys x_axis, metric_key, "algo", "run_id"
        task_data = []
        x_steps = []
        for algo, run_ids in RUN_IDS[benchmark].items():
            all_scores = []
            for run_id in run_ids:
                run = wandb_api.run(f"{ENTITY_AND_PROJECT}/{run_id}")

                history = run.history(
                    samples=999_999_999
                )  # Make sure we get everything
                history = history[[x_axis, metric_key]].dropna()
                env_steps = EPISODE_COUNT * EPISODE_LEN  # Cut off experiment data here
                max_x = history[history[x_axis] >= env_steps][x_axis].iloc[0]
                history = history[history[x_axis] <= max_x]  # Max. 1 data point above
                history["algo"] = algo
                history["run_id"] = run_id
                history = history.sort_values(by=x_axis)
                all_scores.append(history[metric_key].values)
                if len(x_steps) > 0:  # Not the first iter
                    assert len(x_steps) == len(history[x_axis])
                x_steps = history[x_axis]

            scores = np.array(all_scores)  # Shape: (num_seeds, num_steps)
            scores_per_step = {}
            x_keys = [str(x) for x in range(len(x_steps))]
            for x, x_key in enumerate(x_keys):
                scores_per_step[x_key] = scores[:, x].reshape(-1, 1)

            # Get 95% bootstrapped CIs
            aggregate_scores, aggregate_cis = rly.get_interval_estimates(
                scores_per_step,
                aggregate_func,
                reps=RLIABLE_REPS,
                confidence_interval_size=0.95,
            )

            history["mean"] = np.array(
                [aggregate_scores[x] for x in x_keys], dtype=float
            )
            history["lower"] = np.array(
                [aggregate_cis[x][0] for x in x_keys], dtype=float
            )
            history["upper"] = np.array(
                [aggregate_cis[x][1] for x in x_keys], dtype=float
            )

            task_data.append(history)

        # Mush everything together into a pd dataframe
        task_dfs.append(pd.concat(task_data))

    # Plot per-task plots next to each other
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(
        1,
        max_n_tasks,
        figsize=(4 * max_n_tasks, 4),
        sharey=True,
        constrained_layout=True,
    )
    ylim_wishes = []
    for i in range(n_tasks):
        task_df = task_dfs[i]
        ax = axes[i]
        x_axis = f"eval/.{tasks[i]}.env_step"

        for algo in algos:
            color = COLORS[algo]
            df = task_df[task_df["algo"] == algo]
            ax.plot(df[x_axis], df["mean"], label=algo, color=color)
            ax.fill_between(
                df[x_axis], df["lower"], df["upper"], alpha=0.2, color=color
            )

        ax.set_title(tasks[i].capitalize(), fontsize=FONT_SIZE_TITLE)
        if i == 0:
            ax.set_ylabel("Episodic return", fontsize=FONT_SIZE)
        ax.set_xlabel("Environment step", fontsize=FONT_SIZE)

        ax.tick_params(axis="both", labelsize=FONT_SIZE)
        ax.xaxis.set_ticks([0, 200000, 400000, 600000, 800000, 1000000])
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(millions))
        ax.set_xlim(0, task_df[x_axis].max())
        ylim_wishes.append(task_df["upper"].max() * 1.05)  # Add a 5% top-margin

    for i in range(n_tasks, max_n_tasks):
        axes[i].axis("off")  # Hide the empty subplot

    plt.ylim(0, min(EPISODE_LEN, max(ylim_wishes)))

    # Legend inside the first subplot (bottom right, vertical)
    handles, labels = axes[0].get_legend_handles_labels()
    axes[0].legend(handles, labels, loc="lower right", prop={"size": FONT_SIZE})

    # Some magic numbers to make the suptitle visible
    # plt.suptitle(f"Training curve for {benchmark}", y=0.95)
    # plt.tight_layout(rect=(0, 0, 1, 0.98))

    # Save plot
    plt.savefig(
        os.path.join(OUT_DIR, EXPERIMENT, f"curve_{benchmark}.pdf"),
        format="pdf",
    )
