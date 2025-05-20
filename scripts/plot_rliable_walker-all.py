import os

import wandb
import numpy as np
from rliable import plot_utils
import rliable.metrics as metrics
import rliable.library as rly
import pandas as pd


# === Constants ===
ENTITY_AND_PROJECT = "kielen1-aalto-university/iqrl"
BENCHMARK = "walker-all"
OUT_DIR = "results"
os.makedirs(os.path.join(OUT_DIR, BENCHMARK), exist_ok=True)

RLIABLE_REPS = 50000
MAX_RETURN = 1000  # Final ablation study has only half the amount of env steps
XLABEL_Y_COORD_MEAN = -0.7
XLABEL_Y_COORD_PER_TASK = -0.4

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
# Give consistent colors to MultiQRL and the three baseline methods
colors = {
    MultiQRL: COLORS15[0],
    TDMPC2: COLORS15[1],
    OneHotTD3: COLORS15[2],
    iQRL: COLORS15[3],
}

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
    df.to_csv(os.path.join(OUT_DIR, BENCHMARK, f"{filename}.csv"), index=False)


# === Nested metric names (for wandb) & their descriptions (for plot) ===
mt_metric = (
    ("eval/", "episodic_return_mean"),
    f"Normalized Mean Return Across '{BENCHMARK}' Envs",
)
st_metrics = [
    (("eval/", "walker-stand", "episodic_return"), "walker-stand"),
    (("eval/", "walker-walk", "episodic_return"), "walker-walk"),
    (("eval/", "walker-run", "episodic_return"), "walker-run"),
]

# === Specify wandb run IDs and their respective algo descriptions ===
mt_algos = [
    MultiQRL,  # A
    OneHotTD3,  #  Baseline
    # TDMPC2,  #  Baseline TODO
]
mt_run_ids = {
    mt_algos[0]: ["1zp9zqip", "ra7b5sr3", "aenmg9ig", "7150h1oc", "v5wkvaut"],
    mt_algos[1]: ["ef5evw0d", "llpqc46y", "n7owite3", "j2y6k2hk", "wi7fyi1o"],
}

st_algos = [
    f"{MultiQRL} (single-task)",
    iQRL,  # ST baseline
]
st_run_ids = {
    st_algos[0]: {
        "walker-stand": ["dghit6c0", "lopkc49l", "ak7l3smc", "gtabpwxe", "up7goh94"],
        "walker-walk": ["9l04z9e1", "82tld3oh", "esezzowl", "9873okdm", "d4uf4kyw"],
        "walker-run": ["03l0vhnt", "8sf0hjtq", "pk623zcg", "buiaawwe", "bhkax87d"],
    },
    st_algos[1]: {
        "walker-stand": ["m9xgn6ot", "quxs0m3z", "vp38t01e", "684of0b1", "efk0tqo8"],
        "walker-walk": ["nd2caq2a", "z4ayjomc", "k5t3gwqa", "i0k4p0td", "krb69zpi"],
        "walker-run": ["y9w66o50", "kiyghn5z", "m19by8vc", "2bl5k6ss", "ru475g9n"],
    },
}

# Determine remaining colors (MultiQRL and baselines are set as constants above)
color_idx = 0
for algo_name in mt_algos + st_algos:
    if algo_name not in colors:
        while COLORS15[color_idx] in colors.values():
            color_idx += 1
        colors[algo_name] = COLORS15[color_idx]  # Assign next available color

# === Fetch and normalize ===
mt_metric_full, mt_metric_short = mt_metric
raw_scores = collect_scores(mt_run_ids, mt_metric_full)
normalized_scores = {k: v / MAX_RETURN for k, v in raw_scores.items()}

# === Compute aggregate metrics ===
rliable_metric_names = ["IQM", "Mean", "Median", "Optimality Gap"]
aggregate_scores, aggregate_cis = rly.get_interval_estimates(
    normalized_scores, aggregate_func_all, reps=RLIABLE_REPS
)
save_to_csv(
    aggregate_scores,
    aggregate_cis,
    algos=mt_algos,
    rliable_metrics=rliable_metric_names,
    filename=f"{BENCHMARK}_mean",
)

# === Plot ===
fig, axes = plot_utils.plot_interval_estimates(
    aggregate_scores,
    aggregate_cis,
    metric_names=rliable_metric_names,
    algorithms=mt_algos[::-1],  # Show in "correct" order
    xlabel_y_coordinate=XLABEL_Y_COORD_MEAN,
    xlabel=mt_metric_short,
    colors=colors,
)

os.makedirs(os.path.join(OUT_DIR, BENCHMARK), exist_ok=True)
save_fig(fig, os.path.join(OUT_DIR, BENCHMARK, f"{BENCHMARK}_mean"))

# === Create one plot with task-specific IQM side by side ===
# Put mt_algos and st_algos into a sensible order
all_algos = [MultiQRL, f"{MultiQRL} (single-task)", iQRL, OneHotTD3]
per_task_metric_names = []
aggregate_scores = {}
aggregate_cis = {}

for st_metric_full, st_metric_short in st_metrics:
    # Fetch and normalize
    all_run_ids = {
        **mt_run_ids,
        **{st_algo: st_run_ids[st_algo][st_metric_short] for st_algo in st_algos},
    }
    raw_scores = collect_scores(all_run_ids, st_metric_full)
    normalized_scores = {k: v / MAX_RETURN for k, v in raw_scores.items()}

    # Compute aggregate metrics
    aggregate_scores_task, aggregate_cis_task = rly.get_interval_estimates(
        normalized_scores, aggregate_func_iqm, reps=RLIABLE_REPS
    )
    save_to_csv(
        aggregate_scores_task,
        aggregate_cis_task,
        algos=[*mt_algos, *st_algos],
        rliable_metrics=["IQM"],
        filename=f"{BENCHMARK}_per-task_{st_metric_short}",
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

# === Plot per-task IQM ===
fig, axes = plot_utils.plot_interval_estimates(
    aggregate_scores,
    aggregate_cis,
    metric_names=per_task_metric_names,
    algorithms=all_algos[::-1],  # Show in "correct" order
    xlabel_y_coordinate=XLABEL_Y_COORD_PER_TASK,
    xlabel="Normalized Return",
    colors=colors,
)
save_fig(fig, os.path.join(OUT_DIR, BENCHMARK, f"{BENCHMARK}_per-task"))
