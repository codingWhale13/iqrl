import os

import wandb
import numpy as np
from rliable import plot_utils
import rliable.metrics as metrics
import rliable.library as rly
import pandas as pd
import matplotlib.ticker as mticker

# === Constants ===
ENTITY_AND_PROJECT = "kielen1-aalto-university/iqrl"
EXPERIMENT = "exp4"
BENCHMARK = "Fortunate-5"
OUT_DIR = "results"
os.makedirs(os.path.join(OUT_DIR, EXPERIMENT), exist_ok=True)

RLIABLE_REPS = 50000
MAX_RETURN = 1000
XLABEL_Y_COORD = -0.4

# Give consistent names to MultiQRL and the baseline methods
MultiQRL = "MultiQRL"
OneHotTD3 = "MT-TD3"
TDMPC2 = "TD-MPC2 w/o MPC"
SAVE_CSV = False

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

# === Helpers ===
wandb_api = wandb.Api()


def save_fig(fig, name):
    file_name = "{}.pdf".format(name)
    fig.savefig(file_name, format="pdf", bbox_inches="tight")


def fetch_final_metric(run_path, nested_metric):
    # Account for incosistency in naming of logs
    # iQRL: "eval/" -> "episodic_return_mean"
    # TD-MPC2: "eval/episodic_return_mean"
    run = wandb_api.run(os.path.join(ENTITY_AND_PROJECT, run_path))
    final_score = run.summary
    i = 0
    while i < len(nested_metric):
        key_name = nested_metric[i]
        i += 1
        while key_name not in final_score:
            key_name += f".{nested_metric[i]}"
            i += 1
        final_score = final_score[key_name]

    return final_score


def collect_scores(ids, nested_metric):
    score_dict = {}
    for label, run_paths in ids.items():
        values = []
        for path in run_paths:
            val = fetch_final_metric(path, nested_metric)
            if val is not None:
                values.append(val)
        values = np.array(values)
        if len(values) > 0:
            # Reshape to (num_runs, num_tasks=1)
            score_dict[label] = values.reshape(-1, 1)
            print(f"{label}: {len(values)} valid runs")
    return score_dict


def aggregate_func_all(x):
    return np.array(
        [
            metrics.aggregate_iqm(x),
            metrics.aggregate_mean(x),
            metrics.aggregate_median(x),
            metrics.aggregate_optimality_gap(x, gamma=1),
        ]
    )


def aggregate_func_iqm(x):
    return np.array([metrics.aggregate_iqm(x)])


def save_to_csv(aggregate_scores, aggregate_cis, algos, rliable_metrics, filename):
    rows = []
    for i, rliable_metric in enumerate(rliable_metrics):
        for algo in algos:
            score = aggregate_scores[algo][i]
            rows.append(
                {
                    "Metric": rliable_metric,
                    "Experiment": algo,
                    "Score": np.round(score, 3),
                    "CI_low": np.round(aggregate_cis[algo][0][i], 3),
                    "CI_high": np.round(aggregate_cis[algo][1][i], 3),
                }
            )
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT_DIR, EXPERIMENT, f"{filename}.csv"), index=False)


# === Nested metric names (for wandb) & their descriptions (for plot) ===
mt_metric = "eval/", "episodic_return_mean"
st_metrics = [
    (("eval/", "walker-stand", "episodic_return"), "walker-stand"),
    (("eval/", "walker-walk", "episodic_return"), "walker-walk"),
    (("eval/", "walker-run", "episodic_return"), "walker-run"),
    (("eval/", "hopper-stand", "episodic_return"), "hopper-stand"),
    (("eval/", "dog-stand", "episodic_return"), "dog-stand"),
]

# === Specify wandb run IDs and their respective algo descriptions ===
mt_algos = [
    MultiQRL,  # A
    "W/o body info",  # C
    "W/ one-hot",  # D
    "W/ one-hot & w/o body info",  # E
]
mt_run_ids = {
    mt_algos[0]: ["eqg4d9zz", "joigd2cd", "41vruqcy", "aod9te08", "apgkkvmz"],
    mt_algos[1]: ["8e2j0chq", "6ulhitro", "h9onipwz", "owo0ly4e", "upzki7zy"],
    mt_algos[2]: ["4a7llm33", "8lkmp881", "figk8wfn", "hv2at86v", "v34bn00v"],
    mt_algos[3]: ["w6yx7zyb", "61pkw024", "d29nij71", "r9s9cxb5", "rn2n337a"],
}

iqrl_ids = {
    st_metrics[0][1]: ["m9xgn6ot", "quxs0m3z", "vp38t01e", "684of0b1", "efk0tqo8"],
    st_metrics[1][1]: ["nd2caq2a", "z4ayjomc", "k5t3gwqa", "i0k4p0td", "krb69zpi"],
    st_metrics[2][1]: ["y9w66o50", "kiyghn5z", "m19by8vc", "2bl5k6ss", "ru475g9n"],
    st_metrics[3][1]: ["b4hm7ugz", "634tq813", "btfjnh5i", "v391aepi", "av4glxbg"],
    st_metrics[4][1]: ["7ecz7ql9", "s5yptf53", "w587ghla", "efgampel", "tffqryy9"],
}
# Order metrics to emphasize the property of 3x walk, 3x stand
st_metrics = [
    (("eval/", "walker-run", "episodic_return"), "walker-run"),
    (("eval/", "walker-walk", "episodic_return"), "walker-walk"),
    (("eval/", "walker-stand", "episodic_return"), "walker-stand"),
    (("eval/", "hopper-stand", "episodic_return"), "hopper-stand"),
    (("eval/", "dog-stand", "episodic_return"), "dog-stand"),
]

# === Fetch and normalize ===
raw_scores = collect_scores(mt_run_ids, mt_metric)
normalized_scores = {k: v / MAX_RETURN for k, v in raw_scores.items()}

# === Compute aggregate metrics ===
rliable_metric_names = ["IQM", "Mean", "Median", "Optimality Gap"]
aggregate_scores, aggregate_cis = rly.get_interval_estimates(
    normalized_scores, aggregate_func_all, reps=RLIABLE_REPS
)
if SAVE_CSV:
    save_to_csv(
        aggregate_scores,
        aggregate_cis,
        algos=mt_algos,
        rliable_metrics=rliable_metric_names,
        filename=f"{EXPERIMENT}_{BENCHMARK}_mean",
    )

# === Plot ===
fig, axes = plot_utils.plot_interval_estimates(
    aggregate_scores,
    aggregate_cis,
    metric_names=rliable_metric_names,
    algorithms=mt_algos[::-1],  # Show in "correct" order
    xlabel_y_coordinate=XLABEL_Y_COORD,
    xlabel=f"Normalized Mean Return Across {BENCHMARK} Environments",
    colors=colors,
)
for ax in axes:
    ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=3))  # Avoid overlap of plots

save_fig(fig, os.path.join(OUT_DIR, EXPERIMENT, f"{EXPERIMENT}_{BENCHMARK}_mean"))

# === Create one plot with task-specific IQM side by side ===
all_algos = mt_algos
per_task_metric_names = []
aggregate_scores = {}
aggregate_cis = {}

for st_metric_full, st_metric_short in st_metrics:
    # Fetch and normalize
    raw_scores = collect_scores(mt_run_ids, st_metric_full)
    normalized_scores = {k: v / MAX_RETURN for k, v in raw_scores.items()}

    # Compute aggregate metrics
    aggregate_scores_task, aggregate_cis_task = rly.get_interval_estimates(
        normalized_scores, aggregate_func_iqm, reps=RLIABLE_REPS
    )
    if SAVE_CSV:
        save_to_csv(
            aggregate_scores_task,
            aggregate_cis_task,
            algos=all_algos,
            rliable_metrics=["IQM"],
            filename=f"{EXPERIMENT}_{BENCHMARK}_per-task_{st_metric_short}",
        )

    per_task_metric_names.append(f"IQM {st_metric_short}")
    if len(aggregate_scores.keys()) == 0:
        for k in aggregate_scores_task.keys():
            aggregate_scores[k] = [aggregate_scores_task[k]]
            aggregate_cis[k] = [[aggregate_cis_task[k][0]], [aggregate_cis_task[k][1]]]
    else:
        for k in aggregate_scores.keys():
            aggregate_scores[k].append(aggregate_scores_task[k])
            aggregate_cis[k][0].append(aggregate_cis_task[k][0])
            aggregate_cis[k][1].append(aggregate_cis_task[k][1])

for k in aggregate_scores.keys():
    aggregate_scores[k] = np.array(aggregate_scores[k])
    aggregate_cis[k] = np.array(aggregate_cis[k])

fig, axes = plot_utils.plot_interval_estimates(
    aggregate_scores,
    aggregate_cis,
    metric_names=per_task_metric_names,
    algorithms=all_algos[::-1],  # Show in "correct" order
    xlabel_y_coordinate=XLABEL_Y_COORD,
    xlabel=f"Normalized Return per {BENCHMARK} Environment",
    colors=colors,
)

# Indicate "optimal" value (max iQRL return)
for i, ax in enumerate(axes):
    metric_key = ".".join(st_metrics[i][0])
    task = st_metrics[i][1]
    max_score = 0
    for run_id in iqrl_ids[task]:
        run = wandb_api.run(f"{ENTITY_AND_PROJECT}/{run_id}")
        history = run.history(samples=9999)  # Make sure we get everything
        max_score = max(max_score, history[metric_key].dropna().max())
    max_score /= MAX_RETURN
    ax.axvline(x=max_score, color="black", linewidth=1.5, linestyle=":")
    ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=3))  # Avoid overlap of plots

save_fig(fig, os.path.join(OUT_DIR, EXPERIMENT, f"{EXPERIMENT}_{BENCHMARK}_per-task"))
