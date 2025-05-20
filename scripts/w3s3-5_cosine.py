import wandb
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import seaborn as sns
import rliable.library as rly
import rliable.metrics as metrics
import numpy as np
import os

# CONSTANTS
entity_and_project = "kielen1-aalto-university/iqrl"
MAX_SCORE = 1000
REPS = 50_000
OUT_DIR = "results"

benchmark = "fsdw"
tasks = ["finger-spin", "dog-walk"]
n_tasks = len(tasks)

# Up-to-date :D (4am)
algorithms = ["MultiQRL", "w/ env-encoding (1,2,3,4,5)", "MultiQRL 1-hot", "MultIQRL 1-hot w/ env-encoding"]
ids = {
    algorithms[0]: ["eqg4d9zz", "joigd2cd", "41vruqcy", "aod9te08", "apgkkvmz"],
    algorithms[1]: ["6ulhitro", "h9onipwz", "owo0ly4e", "upzki7zy", "8e2j0chq"],
    algorithms[2]: ["4a7llm33", "8lkmp881", "figk8wfn", "hv2at86v", "v34bn00v"],
    algorithms[3]: ["w6yx7zyb", "61pkw024", "d29nij71", "r9s9cxb5", "rn2n337a"],
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


# --- CONFIG --- #
# Get data from W&B and compute confidence intervals
task_dfs = []
for task in tasks:
    metric_key = f"eval/.{task}.episodic_return"
    x_axis = f"eval/.{task}.env_step"

    # --- SCRIPT --- #
    api = wandb.Api()

    # Elements will be DataFrame with keys x_axis, metric_key, "algo", "run_id"
    task_data = []
    x_steps = []
    for algo, run_ids in ids.items():
        all_scores = []
        for run_id in run_ids:
            run = api.run(f"{entity_and_project}/{run_id}")
            history = run.history(samples=9999)  # Make sure we get everything
            history = history[[x_axis, metric_key]].dropna()
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
            reps=REPS,
            confidence_interval_size=0.95,
        )

        history["mean"] = np.array([aggregate_scores[x] for x in x_keys], dtype=float)
        history["lower"] = np.array([aggregate_cis[x][0] for x in x_keys], dtype=float)
        history["upper"] = np.array([aggregate_cis[x][1] for x in x_keys], dtype=float)

        task_data.append(history)

    # Mush everything together into a pd dataframe
    task_dfs.append(pd.concat(task_data))

# Plot three plots next to each other
sns.set_theme(style="whitegrid")
fig, axes = plt.subplots(1, n_tasks, figsize=(4 * n_tasks, 4), sharey=True)
ylim_wishes = []
for i in range(n_tasks):
    task_df = task_dfs[i]
    ax = axes[i]
    x_axis = f"eval/.{tasks[i]}.env_step"

    for algo, df in task_df.groupby("algo"):
        ax.plot(df[x_axis], df["mean"], label=algo)
        ax.fill_between(df[x_axis], df["lower"], df["upper"], alpha=0.2)

    ax.set_title(tasks[i])
    if i == 0:
        ax.set_ylabel("Episodic Return")
    ax.set_xlabel("Environment Step")

    ax.xaxis.set_major_formatter(mticker.FuncFormatter(millions))
    ax.set_xlim(0, task_df[x_axis].max())
    ylim_wishes.append(task_df["upper"].max() * 1.05)  # Add a 5% top-margin

plt.ylim(0, min(MAX_SCORE, max(ylim_wishes)))

# Legend inside the first subplot (bottom right, vertical)
handles, labels = axes[0].get_legend_handles_labels()
axes[0].legend(
    handles,
    labels,
    loc="lower right",  # position inside the first plot
    fontsize="medium",
    frameon=True,
    ncol=1,  # stacked vertically
    borderpad=0.5,
    handlelength=1.5,
)


plt.suptitle("Performance across tasks with 95% bootstrapped CIs", y=1.05)
plt.tight_layout()
# plt.subplots_adjust(bottom=0, top=1)  # Make room for legend
os.makedirs(os.path.join(OUT_DIR, benchmark), exist_ok=True)
plt.savefig(os.path.join(OUT_DIR, benchmark, f"{benchmark}_curve.pdf"), format="pdf")
