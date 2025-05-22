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
EXPERIMENT = "exp2_exp3"
OUT_DIR = "results"
os.makedirs(os.path.join(OUT_DIR, EXPERIMENT), exist_ok=True)

BENCHMARKS_ALL = ["Tricky-3", "Walker-All", "Spin-Dog", "Fortunate-5", "Final-4"]
RLIABLE_REPS = 50000
MAX_RETURN_ALL = {
    benchmark: 500 if benchmark == "Tricky-3" else 1000 for benchmark in BENCHMARKS_ALL
}
XLABEL_Y_COORD_PER_TASK = -0.5
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
mt_metric = ("eval/", "episodic_return_mean")
ST_METRICS_ALL = {
    BENCHMARKS_ALL[0]: [
        (("eval/", "finger-spin", "episodic_return"), "finger-spin"),
        (("eval/", "pendulum-swingup", "episodic_return"), "pendulum-swingup"),
        (("eval/", "hopper-stand", "episodic_return"), "hopper-stand"),
    ],
    BENCHMARKS_ALL[1]: [
        (("eval/", "walker-stand", "episodic_return"), "walker-stand"),
        (("eval/", "walker-walk", "episodic_return"), "walker-walk"),
        (("eval/", "walker-run", "episodic_return"), "walker-run"),
    ],
    BENCHMARKS_ALL[2]: [
        (("eval/", "finger-spin", "episodic_return"), "finger-spin"),
        (("eval/", "dog-walk", "episodic_return"), "dog-walk"),
    ],
    BENCHMARKS_ALL[3]: [
        (("eval/", "walker-stand", "episodic_return"), "walker-stand"),
        (("eval/", "walker-walk", "episodic_return"), "walker-walk"),
        (("eval/", "walker-run", "episodic_return"), "walker-run"),
        (("eval/", "hopper-stand", "episodic_return"), "hopper-stand"),
        (("eval/", "dog-stand", "episodic_return"), "dog-stand"),
    ],
    BENCHMARKS_ALL[4]: [
        (("eval/", "cartpole-balance", "episodic_return"), "cartpole-balance"),
        (("eval/", "cup-catch", "episodic_return"), "cup-catch"),
        (("eval/", "finger-spin", "episodic_return"), "finger-spin"),
        (("eval/", "pendulum-swingup", "episodic_return"), "pendulum-swingup"),
    ],
}


# === Specify W&B Run IDs===
mt_algos = ["MultiQRL", "MT-TD3", "TD-MPC2 w/o MPC"]
MT_RUN_IDS_ALL = {
    BENCHMARKS_ALL[0]: {
        mt_algos[0]: ["c9jkk4o9", "fdtmlupm", "hxo9e892", "f6fky7wj", "eonoz2ng"],
        mt_algos[1]: ["23hhsr8s", "9jgtsrkk", "q9ihnejt", "xd3e32hf", "xiwp41iu"],
        mt_algos[2]: ["pt4qtojc", "zsngc1mx", "rova5e9n", "mhpkeet9", "opp6nf56"],
    },
    BENCHMARKS_ALL[1]: {
        mt_algos[0]: ["1zp9zqip", "ra7b5sr3", "aenmg9ig", "7150h1oc", "v5wkvaut"],
        mt_algos[1]: ["ef5evw0d", "llpqc46y", "n7owite3", "j2y6k2hk", "wi7fyi1o"],
        mt_algos[2]: ["ueufardu", "d2h05vbt", "svwvvk2w", "z2x27zqf", "8s5vfiwa"],
    },
    BENCHMARKS_ALL[2]: {
        mt_algos[0]: ["me4qpu7w", "lqd17o0w", "nbwt66h6", "h3n98fmj", "shuxbe3r"],
        mt_algos[1]: ["2y76stwi", "bis3oe6h", "dpgyng52", "hfksy4km", "nuk8x7nh"],
        mt_algos[2]: ["ye2ldolb", "g2zxgbgy", "ud6uothg", "a0jn1j7q", "d2iaolgf"],
    },
    BENCHMARKS_ALL[3]: {
        mt_algos[0]: ["eqg4d9zz", "joigd2cd", "41vruqcy", "aod9te08", "apgkkvmz"],
        mt_algos[1]: ["ggrmi8z8", "g8mkd0h8", "3t33di5n", "n0qm2wcn", "sfdgi95m"],
        mt_algos[2]: ["atk38eck", "4crzssmy", "z0f0wnxn", "1bpsvau6", "z4rkd271"],
    },
    BENCHMARKS_ALL[4]: {
        mt_algos[0]: ["qduq3258", "0ohltkcw", "15t6dphm", "78vm2o3k", "yqk11i6w"],
        mt_algos[1]: ["omgie5ti", "28gu5eye", "5ib9yw44", "jyjx1vdw", "q21pje76"],
        mt_algos[2]: ["fq7n36l3", "qx2f40xt", "9bfd9ajd", "dyutz0o9", "ng2ieem0"],
    },
}

IQRL_IDS_ALL = {
    BENCHMARKS_ALL[0]: {
        "finger-spin": ["i9c7czoz", "9m07dyrm", "rd9drmlu", "b9n8detd", "up2nsn2v"],
        "pendulum-swingup": [
            "7yzfo6ap",
            "sr9ny6qq",
            "ef835602",
            "is38syrr",
            "k715goj6",
        ],
        "hopper-stand": ["pvdnwltq", "3meebuoy", "uy97vece", "qdgwozb8", "g36vj3o5"],
    },
    BENCHMARKS_ALL[1]: {
        "walker-stand": ["m9xgn6ot", "quxs0m3z", "vp38t01e", "684of0b1", "efk0tqo8"],
        "walker-walk": ["nd2caq2a", "z4ayjomc", "k5t3gwqa", "i0k4p0td", "krb69zpi"],
        "walker-run": ["y9w66o50", "kiyghn5z", "m19by8vc", "2bl5k6ss", "ru475g9n"],
    },
    BENCHMARKS_ALL[2]: {
        "finger-spin": ["m2n49gjo", "4yuovip0", "apy3vrjd", "g68uxlvq", "qlslbr5c"],
        "dog-walk": ["9qfew33i", "b5qpk0y6", "wm78qp2k", "8p539rl8", "t4tg0sos"],
    },
    BENCHMARKS_ALL[3]: {
        "walker-stand": ["m9xgn6ot", "quxs0m3z", "vp38t01e", "684of0b1", "efk0tqo8"],
        "walker-walk": ["nd2caq2a", "z4ayjomc", "k5t3gwqa", "i0k4p0td", "krb69zpi"],
        "walker-run": ["y9w66o50", "kiyghn5z", "m19by8vc", "2bl5k6ss", "ru475g9n"],
        "hopper-stand": ["b4hm7ugz", "634tq813", "btfjnh5i", "v391aepi", "av4glxbg"],
        "dog-stand": ["7ecz7ql9", "s5yptf53", "w587ghla", "efgampel", "tffqryy9"],
    },
    BENCHMARKS_ALL[4]: {
        "cartpole-balance": [
            "6vulo89q",
            "my4zf30e",
            "0pmnvezl",
            "27pgwdde",
            "fxjknclp",
        ],
        "cup-catch": ["2rsz5dbs", "6ofo8jn3", "f7am5zzr", "j6r55wdp", "v7wsizsz"],
        "finger-spin": ["1mjfvf08", "hzct25nc", "nh7kl3zn", "pkhw7xx5", "rqj76w8y"],
        "pendulum-swingup": [
            "a9x80j30",
            "uk1e5id0",
            "70obpjwj",
            "tctljema",
            "x2eunga6",
        ],
    },
}

# Now is a good time to re-order the benchmarks (consistently across all thesis plots!)
BENCHMARKS_ALL = ["Walker-All", "Spin-Dog", "Fortunate-5", "Tricky-3", "Final-4"]

# Calculate the IQMs for each benchmark and create per-task plots right away
mt_aggregate_scores = {}
mt_aggregate_cis = {}
for benchmark in BENCHMARKS_ALL:
    st_metrics = ST_METRICS_ALL[benchmark]
    max_return = MAX_RETURN_ALL[benchmark]
    mt_run_ids = MT_RUN_IDS_ALL[benchmark]
    iqrl_ids = IQRL_IDS_ALL[benchmark]

    # === Fetch and normalize ===
    raw_scores = collect_scores(mt_run_ids, mt_metric)
    normalized_scores = {k: v / max_return for k, v in raw_scores.items()}

    # === Compute aggregate metrics ===
    aggregate_scores, aggregate_cis = rly.get_interval_estimates(
        normalized_scores, aggregate_func_iqm, reps=RLIABLE_REPS
    )
    if SAVE_CSV:
        save_to_csv(
            aggregate_scores,
            aggregate_cis,
            algos=mt_algos,
            rliable_metrics=["IQM"],
            filename=f"{EXPERIMENT}_{benchmark}_mean",
        )

    # Save MT results for later (want to plot benchmarks side by side)
    if len(mt_aggregate_scores.keys()) == 0:
        for k in aggregate_scores.keys():
            mt_aggregate_scores[k] = [aggregate_scores[k]]
            mt_aggregate_cis[k] = [
                [aggregate_cis[k][0]],
                [aggregate_cis[k][1]],
            ]
    else:
        for k in aggregate_scores.keys():
            mt_aggregate_scores[k].append(aggregate_scores[k])
            mt_aggregate_cis[k][0].append(aggregate_cis[k][0])
            mt_aggregate_cis[k][1].append(aggregate_cis[k][1])

    # === Create one plot with task-specific IQM side by side ===
    per_task_metric_names = []
    st_aggregate_scores = {}
    st_aggregate_cis = {}

    for st_metric_full, st_metric_short in st_metrics:
        # Fetch and normalize
        raw_scores = collect_scores(mt_run_ids, st_metric_full)
        normalized_scores = {k: v / max_return for k, v in raw_scores.items()}

        # Compute aggregate metrics
        aggregate_scores_task, aggregate_cis_task = rly.get_interval_estimates(
            normalized_scores, aggregate_func_iqm, reps=RLIABLE_REPS
        )
        if SAVE_CSV:
            save_to_csv(
                aggregate_scores_task,
                aggregate_cis_task,
                algos=mt_algos,
                rliable_metrics=["IQM"],
                filename=f"{EXPERIMENT}_{benchmark}_per-task_{st_metric_short}",
            )

        per_task_metric_names.append(f"IQM {st_metric_short}")
        if len(st_aggregate_scores.keys()) == 0:
            for k in aggregate_scores_task.keys():
                st_aggregate_scores[k] = [aggregate_scores_task[k]]
                st_aggregate_cis[k] = [
                    [aggregate_cis_task[k][0]],
                    [aggregate_cis_task[k][1]],
                ]
        else:
            for k in aggregate_scores_task.keys():
                st_aggregate_scores[k].append(aggregate_scores_task[k])
                st_aggregate_cis[k][0].append(aggregate_cis_task[k][0])
                st_aggregate_cis[k][1].append(aggregate_cis_task[k][1])

    for k in aggregate_scores.keys():
        st_aggregate_scores[k] = np.array(st_aggregate_scores[k])
        st_aggregate_cis[k] = np.array(st_aggregate_cis[k])

    # === Plot per-task IQM ===
    fig, axes = plot_utils.plot_interval_estimates(
        st_aggregate_scores,
        st_aggregate_cis,
        metric_names=per_task_metric_names,
        algorithms=mt_algos[::-1],  # Show in "correct" order
        xlabel_y_coordinate=XLABEL_Y_COORD_PER_TASK,
        xlabel=f"Normalized Return per {benchmark} Environment",
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
        max_score /= max_return
        ax.axvline(x=max_score, color="black", linewidth=1.5, linestyle=":")
        ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=3))  # Avoid plot overlap

    save_fig(
        fig, os.path.join(OUT_DIR, EXPERIMENT, f"{EXPERIMENT}_{benchmark}_per-task")
    )
