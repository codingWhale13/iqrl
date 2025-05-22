import wandb
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.lines as mlines
import numpy as np
import seaborn as sns
import rliable.library as rly
import rliable.metrics as metrics
import numpy as np
import os

# CONSTANTS
entity_and_project = "kielen1-aalto-university/iqrl"
MAX_SCORE = 1000
REPS = 50000
OUT_DIR = "results"
EXPERIMENT = "exp1"
os.makedirs(os.path.join(OUT_DIR, EXPERIMENT), exist_ok=True)

BENCHMARK = "walker-all"
tasks = ["walker-stand", "walker-walk", "walker-run"]
n_tasks = len(tasks)

# Map consistently from names to colors from https://sashamaps.net/docs/resources/20-colors/
COLORS15_RGB = {
    # MultiQRL & baselines
    "MultiQRL": (230, 25, 75),  # Red
    "MultiQRL (single-task)": (128, 128, 0),  # Olive
    "MT-TD3": (245, 130, 48),  # Orange
    "TD-MPC2 w/o MPC": (0, 130, 200),  # Blue
    # Ablations
    "W/o CE": (128, 128, 128),  # Gray
    "W/o FSQ & w/o CE": (60, 180, 75),  # Green
    "W/o soft-CE": (200, 150, 255),  # Lavender but made darker -> more visible
    "W/ n-step=1": (0, 128, 128),  # Teal
    "W/ n-step=3": (170, 110, 40),  # Brown
    "W/o reward model": (255, 225, 25),  # Yellow
    "W/o cond. LayerNorm": (250, 150, 190),  # Pink but made darker -> more visible
    "Cond. only encoder": (210, 245, 60),  # Lime
    "Cond. only 1st layer": (48, 48, 255),  # Navy but made brighter -> contrast w/ mean
    "W/ one-hot": (240, 50, 230),  # Magenta
    "W/o body info": (128, 0, 0),  # Maroon
    "W/ one-hot & w/o body info": (145, 30, 180),  # Purple
}
colors = {k: (r / 255, g / 255, b / 255) for k, (r, g, b) in COLORS15_RGB.items()}

mt_algos = ["MultiQRL"]
mt_run_ids = {mt_algos[0]: ["1zp9zqip", "ra7b5sr3", "aenmg9ig", "7150h1oc", "v5wkvaut"]}

st_algos = ["MultiQRL (single-task)"]
st_ids_ws = ["dghit6c0", "lopkc49l", "ak7l3smc", "gtabpwxe", "up7goh94"]
st_ids_ww = ["9l04z9e1", "82tld3oh", "esezzowl", "9873okdm", "d4uf4kyw"]
st_ids_wr = ["03l0vhnt", "8sf0hjtq", "pk623zcg", "buiaawwe", "bhkax87d"]

iqrl_ids = {
    tasks[0]: ["m9xgn6ot", "quxs0m3z", "vp38t01e", "684of0b1", "efk0tqo8"],
    tasks[1]: ["nd2caq2a", "z4ayjomc", "k5t3gwqa", "i0k4p0td", "krb69zpi"],
    tasks[2]: ["y9w66o50", "kiyghn5z", "m19by8vc", "2bl5k6ss", "ru475g9n"],
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
wandb_api = wandb.Api()
task_dfs = []
max_scores_iqrl = []
for task in tasks:
    if task == "walker-stand":
        runs_dict = {**mt_run_ids, st_algos[0]: st_ids_ws}
    elif task == "walker-walk":
        runs_dict = {**mt_run_ids, st_algos[0]: st_ids_ww}
    elif task == "walker-run":
        runs_dict = {**mt_run_ids, st_algos[0]: st_ids_wr}

    metric_key = f"eval/.{task}.episodic_return"
    x_axis = f"eval/.{task}.env_step"

    # --- SCRIPT --- #
    # Determine max iQRL value for this task
    max_score = 0
    for run_id in iqrl_ids[task]:
        run = wandb_api.run(f"{entity_and_project}/{run_id}")
        history = run.history(samples=9999)  # Make sure we get everything
        max_score = max(max_score, history[metric_key].dropna().max())
    max_scores_iqrl.append(max_score)

    # Elements will be DataFrame with keys x_axis, metric_key, "algo", "run_id"
    task_data = []
    x_steps = []
    for algo, run_ids in runs_dict.items():
        all_scores = []
        for run_id in run_ids:
            run = wandb_api.run(f"{entity_and_project}/{run_id}")
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

# Plot per-task plots next to each other
sns.set_theme(style="whitegrid")
fig, axes = plt.subplots(1, n_tasks, figsize=(4 * n_tasks, 4), sharey=True)
ylim_wishes = []
for i in range(n_tasks):
    task_df = task_dfs[i]
    ax = axes[i]
    x_axis = f"eval/.{tasks[i]}.env_step"

    for algo in mt_algos + st_algos:
        color = colors[algo]
        df = task_df[task_df["algo"] == algo]
        ax.plot(df[x_axis], df["mean"], label=algo, color=color)
        ax.fill_between(df[x_axis], df["lower"], df["upper"], alpha=0.2, color=color)
    ax.axhline(y=max_scores_iqrl[i], color="black", linewidth=1.5, linestyle=":")

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
handles.append(mlines.Line2D([], [], color="black", linewidth=1.5, linestyle=":"))
labels.append("Max. iQRL (single-task)")
axes[0].legend(handles, labels, loc="lower right")

# Some magic numbers to make the suptitle visible
plt.suptitle(
    "Multi-Task vs. Single-Task Performance of MultiQRL for {BENCHMARK}", y=0.98
)
plt.tight_layout(rect=(0, 0, 1, 0.98))

# Save plot
plt.savefig(
    os.path.join(OUT_DIR, EXPERIMENT, f"{EXPERIMENT}_{BENCHMARK}_curve.pdf"),
    format="pdf",
)
