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

benchmark = "walker-all"
tasks = ["walker-stand", "walker-walk", "walker-run"]
n_tasks = len(tasks)
# Give consistent names to MultiQRL and the three baseline methods
MultiQRL = "MultiQRL"
TDMPC2 = "TD-MPC2 w/ mpc=False"
OneHotTD3 = "One-hot MT-TD3"
iQRL = "iQRL (single-task)"

# My favorite 15 colors from https://sashamaps.net/docs/resources/20-colors/
COLORS15_RGB = [
    (230, 25, 75),  # Red   => MultiQRL
    (0, 130, 200),  # Blue  => TD-MPC2
    (245, 130, 48),  # Orange => MT-TD3 with one-hot encodings
    (60, 180, 75),  # Green => iQRL
    (0, 128, 128),  # Teal
    (170, 110, 40),  # Brown
    (240, 50, 230),  # Magenta
    (128, 0, 0),  # Maroon
    (0, 0, 128),  # Navy
    (128, 128, 128),  # Gray
    (220, 190, 255),  # Lavender
    (255, 225, 25),  # Yellow
    (250, 190, 212),  # Pink
    (170, 255, 195),  # Mint
    (70, 240, 240),  # Cyan
]
COLORS15 = [(r / 255, g / 255, b / 255) for r, g, b in COLORS15_RGB]
mt_algos = [
    "MultiQRL (trained on walker-all)",
    # OneHotTD3,
]
mt_run_ids = {
    mt_algos[0]: ["1zp9zqip", "ra7b5sr3", "aenmg9ig", "7150h1oc", "v5wkvaut"],
    # mt_algos[1]: ["ef5evw0d", "llpqc46y", "n7owite3", "j2y6k2hk", "wi7fyi1o"],
}

st_algos = ["MultiQRL (single-task)"]
st_ids_ws = ["dghit6c0", "lopkc49l", "ak7l3smc", "gtabpwxe", "up7goh94"]
st_ids_ww = ["9l04z9e1", "82tld3oh", "esezzowl", "9873okdm", "d4uf4kyw"]
st_ids_wr = ["03l0vhnt", "8sf0hjtq", "pk623zcg", "buiaawwe", "bhkax87d"]

# Determine remaining colors (MultiQRL and baselines are set as constants above)
# Set these manually, because MultiQRL name is different in this plot...
# Give consistent colors to MultiQRL and the three baseline methods
colors = {
    mt_algos[0]: COLORS15[0],
    st_algos[0]: COLORS15[4],  # teal, to match rliable plot
    # TDMPC2: COLORS15[1],
    # OneHotTD3: COLORS15[2],
    # iQRL: COLORS15[3],
    # st_algos[0]: COLORS15[9],  # gray, to not distract from the others
}

# Determine remaining colors (MultiQRL and baselines are set as constants above)
color_idx = 0
for algo_name in mt_algos + st_algos:
    if algo_name not in colors:
        while COLORS15[color_idx] in colors.values():
            color_idx += 1
        colors[algo_name] = COLORS15[color_idx]  # Assign next available color


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
    if task == "walker-stand":
        runs_dict = {**mt_run_ids, st_algos[0]: st_ids_ws}
    elif task == "walker-walk":
        runs_dict = {**mt_run_ids, st_algos[0]: st_ids_ww}
    elif task == "walker-run":
        runs_dict = {**mt_run_ids, st_algos[0]: st_ids_wr}

    metric_key = f"eval/.{task}.episodic_return"
    x_axis = f"eval/.{task}.env_step"

    # --- SCRIPT --- #
    api = wandb.Api()

    # Elements will be DataFrame with keys x_axis, metric_key, "algo", "run_id"
    task_data = []
    x_steps = []
    for algo, run_ids in runs_dict.items():
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
