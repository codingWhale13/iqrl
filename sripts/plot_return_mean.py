import wandb
import numpy as np
import pandas as pd
from rliable import plot_utils
import rliable.metrics as metrics
import rliable.library as rly
import os


REPS = 50000
OUT_DIR = "results"


# === Helper Functions ===
def save_fig(fig, name):
    file_name = "{}.pdf".format(name)
    fig.savefig(file_name, format="pdf", bbox_inches="tight")


def fetch_final_metric(run_path, metric_name):
    run = wandb.Api().run(f"{entity_and_project}/{run_path}")
    final_score = run.summary

    # Account for incosistency in naming of logs
    # iQRL: "eval/" -> "episodic_return_mean"
    # MultiQRL: "eval/episodic_return_mean"
    i = 0
    while i < len(metric_name):
        key_name = metric_name[i]
        i += 1
        while key_name not in final_score:
            key_name += f".{metric_name[i]}"
            i += 1
        final_score = final_score[key_name]

    return final_score


def collect_scores(ids, metric_name):
    score_dict = {}
    for label, run_paths in ids.items():
        values = []
        for path in run_paths:
            val = fetch_final_metric(path, metric_name)
            if val is not None:
                values.append(val)
        values = np.array(values)
        if len(values) > 0:
            # Reshape to (num_runs, num_tasks=1)
            score_dict[label] = values.reshape(-1, 1)
            print(f"{label}: {len(values)} valid runs")
    return score_dict


# === Nested metric names & Shorthands ===
total_return_name = (
    ("eval/", "episodic_return_mean"),
    "Normalized Return Mean Across Tricky-3 Envs",
)
fs_return_name = (
    ("eval/", "finger-spin", "episodic_return"),
    "finger-spin",
)
ps_return_name = (
    ("eval/", "pendulum-swingup", "episodic_return"),
    "pendulum-swingup",
)
hs_return_name = (
    ("eval/", "hopper-stand", "episodic_return"),
    "hopper-stand",
)

# === Config ===
entity_and_project = "kielen1-aalto-university/iqrl"
benchmark = "tricky-3"  # Toggle between hd-stand, tricky-3-v0, tricky-3, and final-5

t3_algos = [
    "MultiQRL",
    "1-hot",
    "1-hot LayerNorm",
    "1-hot all layers",
    "1-hot all layers + LayerNorm",
    "w/o cross-entropy",
]
t3_ids = {
    t3_algos[0]: ["ex73rfi6", "f367nok6", "xhrse6yp"],
    t3_algos[1]: ["hqnt3hnv", "peqipjxr", "ee7qcz2m"],
    t3_algos[2]: ["pkuw6oxo", "jemrye7j", "yl0qy0hb"],
    t3_algos[3]: ["b5alysls", "ekn0a6hn"],  # NOTE: last seed crashed, thus not included
    t3_algos[4]: ["j872nf4x", "j1siiylz", "vcjjpmpv"],
    t3_algos[5]: ["nut830m0", "z6cnwnm9", "bz8h76ev"],
}

t3_final_algos = [
    "MultiQRL",  # A
    "W/o reward model",  # B
    "W/ n-step=1",  # C
    "W/o CE",  # D
    "W/o soft-CE",  # E
    "W/ one-hot",  # F
    "Cond. only 1st layer",  # G
    "W/o cond. LayerNorm",  # H
    "Cond. only encoder",  # I
    "W/o FSQ (& w/o CE)",  # J
    # "TD-MPC2 w/o MPC",  #  Baseline
]
t3_final_ids = {
    t3_final_algos[0]: ["c9jkk4o9", "fdtmlupm", "hxo9e892", "f6fky7wj", "eonoz2ng"],
    t3_final_algos[1]: ["6vt0wu8y", "s3tfuxk2", "y7cpc3f2", "ka1grhx3", "ghjkvist"],
    t3_final_algos[2]: ["2r5220t6", "3by9fvsa", "fckbdynk", "t6yn7ikb", "djq5oz43"],
    t3_final_algos[3]: ["4p96cwdi", "5xeg7y5j", "ig86y66r", "6w5i99lx", "sa1d6pak"],
    t3_final_algos[4]: ["p83mwvfl", "0skyv29t", "9ufb0krt", "c66y8o2f", "pvh9tmly"],
    t3_final_algos[5]: ["d3r4uuib", "6yi0cufw", "cglz2804", "ycyadrud", "v2tvicny"],
    t3_final_algos[6]: ["b70hf02x", "ojd6risa", "r0nio6ii", "iulebgd0", "9mtxv1up"],
    t3_final_algos[7]: ["1t8ii962", "npmkdtek", "r71ksdyd", "34k3fggm", "e8h5ntuj"],
    t3_final_algos[8]: ["bwyogl52", "ijrsbf9e", "807uir8r", "v1jr6mfw", "k7mscq1y"],
    t3_final_algos[9]: ["f86wwbif", "7862249j", "9x2rt8p2", "oqtf9bgu", "1q9yt9yl"],
    # t3_final_algos[10]: ["pt4qtojc", "zsngc1mx", "rova5e9n", "mhpkeet9", "opp6nf56"],
}
# Now that we got the right pairing between name (letter) and data (seeds)
# Re-order by descending IQM
t3_final_algos = [
    "MultiQRL",  # A (reason: this is our proposed algorithm, put it on top)
    "W/ n-step=1",  # C
    "W/o CE",  # D
    "W/o reward model",  # B
    "W/o cond. LayerNorm",  # H
    "Cond. only encoder",  # I
    "Cond. only 1st layer",  # G
    "W/o soft-CE",  # E
    "W/ one-hot",  # F
    "W/o FSQ (& w/o CE)",  # J
    # "TD-MPC2 w/o MPC",  #  Baseline
]

hds_algos = ["MultiQRL", "w/o cross-entropy"]
hds_ids = {
    hds_algos[0]: ["l90vxafk", "o8frsgoh", "oqm5mm50"],
    hds_algos[1]: ["iqttzjv0", "kdv5sg9g", "knuiz2b4"],
}

f5_algos = ["MultiQRL", "w/o soft-ce", "iQRL features", "w/o FSQ", "w/o cross-entropy"]
f5_ids = {
    f5_algos[0]: [
        "havdmpou",
        "k3jtxbt4",
        "8g8g7dqv",
        "ysldory0",
        "yz5b7llx",
        "wy43gbk6",
    ],
    f5_algos[1]: [
        "bidtv3km",
        "janjc7oy",
        "w8tyzd6b",
        "6ya665im",
        "gy62vw8b",
        "0blu1s19",
    ],
    f5_algos[2]: ["jk2iha0q", "38l43nko"],  # Seeds 1 and 5
    f5_algos[3]: [
        "nx3uy69b",
        "1gl5f735",
        "4n9rpupe",
        "6vd4uwb0",
        "fs3lr8f5",
        "iillxqjj",
    ],
    f5_algos[4]: ["74fr6vdc", "ytrgvkhy", "2y4kvyh3"],  # Seeds 0, 1, 2
}


max_return = 1000
if benchmark == "hd-stand":
    ids = hds_ids
    algorithms = hds_algos
    xlabel_y_coordinate = -0.7
elif benchmark == "tricky-3-old":
    ids = t3_ids
    algorithms = t3_algos
    xlabel_y_coordinate = -0.2
elif benchmark == "tricky-3":
    ids = t3_final_ids
    algorithms = t3_final_algos
    xlabel_y_coordinate = -0.1
    max_return = 500  # Final ablation study has only half the amount of env steps
elif benchmark == "final-5":
    ids = f5_ids
    algorithms = f5_algos
    xlabel_y_coordinate = -0.3
algorithms = algorithms[::-1]  # Print in "correct" order from top to down

os.makedirs(os.path.join(OUT_DIR, benchmark), exist_ok=True)
for metric_name, shorthand in [total_return_name]:
    # === Fetch and normalize ===
    raw_scores = collect_scores(ids, metric_name)
    normalized_scores = {k: v / max_return for k, v in raw_scores.items()}

    # === Compute aggregate metrics ===
    aggregate_func = lambda x: np.array(
        [
            metrics.aggregate_iqm(x),
            metrics.aggregate_mean(x),
            metrics.aggregate_median(x),
            metrics.aggregate_optimality_gap(x, gamma=1),
        ]
    )

    aggregate_scores, aggregate_cis = rly.get_interval_estimates(
        normalized_scores, aggregate_func, reps=REPS
    )

    rliable_metric_names = ["IQM", "Mean", "Median", "Optimality Gap"]

    # === Save to CSV / LaTeX ===
    csv = True
    if csv:
        rows = []
        for i, rliable_metric in enumerate(rliable_metric_names):
            for algo in algorithms:
                score = aggregate_scores[algo][i]
                ci_half_width = (
                    aggregate_cis[algo][1][i] - aggregate_cis[algo][0][i]
                ) / 2
                rows.append(
                    {
                        "Metric": rliable_metric,
                        "Experiment": algo,
                        "Score": np.round(score, 3),
                        "95% CI": f"±{np.round(ci_half_width, 3)}",
                    }
                )

        df = pd.DataFrame(rows)
        df.to_csv(
            os.path.join(OUT_DIR, benchmark, f"{shorthand}.csv"),
            index=False,
        )
        df.to_latex(
            os.path.join(OUT_DIR, benchmark, f"{shorthand}.tex"),
            index=False,
            escape=False,
        )
        # print("\nAggregate metrics:")
        # print(df)

    # === Plot ===
    fig, axes = plot_utils.plot_interval_estimates(
        aggregate_scores,
        aggregate_cis,
        metric_names=rliable_metric_names,
        algorithms=algorithms,
        xlabel_y_coordinate=xlabel_y_coordinate,
        xlabel=shorthand,
    )

    os.makedirs(os.path.join(OUT_DIR, benchmark), exist_ok=True)
    save_fig(fig, os.path.join(OUT_DIR, benchmark, f"{benchmark}_mean"))

# ===============

# Create one plot with task-specific IQM side by side
per_task_metric_names = []
aggregate_scores = {}
aggregate_cis = {}
for metric_name, shorthand in [fs_return_name, ps_return_name, hs_return_name]:
    # === Fetch and normalize ===
    raw_scores = collect_scores(ids, metric_name)
    normalized_scores = {k: v / max_return for k, v in raw_scores.items()}

    # === Compute aggregate metrics ===
    aggregate_func = lambda x: np.array([metrics.aggregate_iqm(x)])
    as_task, ac_task = rly.get_interval_estimates(
        normalized_scores, aggregate_func, reps=REPS
    )
    per_task_metric_names.append(f"IQM {shorthand}")
    if len(aggregate_scores.keys()) == 0:
        for k in as_task.keys():
            aggregate_scores[k] = [as_task[k]]
            aggregate_cis[k] = [[ac_task[k][0]], [ac_task[k][1]]]
    else:
        for k in aggregate_scores.keys():
            aggregate_scores[k].append(as_task[k])
            aggregate_cis[k][0].append(ac_task[k][0])
            aggregate_cis[k][1].append(ac_task[k][1])


for k in aggregate_scores.keys():
    aggregate_scores[k] = np.array(aggregate_scores[k])
    aggregate_cis[k] = np.array(aggregate_cis[k])

# === Plot ===
fig, axes = plot_utils.plot_interval_estimates(
    aggregate_scores,
    aggregate_cis,
    metric_names=per_task_metric_names,
    algorithms=algorithms,
    xlabel_y_coordinate=xlabel_y_coordinate,
    xlabel="Normalized Return",
)

save_fig(fig, os.path.join(OUT_DIR, benchmark, f"{benchmark}_per-task"))
