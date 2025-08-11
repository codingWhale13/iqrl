import os

import numpy as np
import rliable.metrics as metrics
import rliable.library as rly
import wandb
import matplotlib.ticker as mticker
from rliable import plot_utils


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
MAX_ALGO_COUNT = max(len(TASKS[benchmark]) for benchmark in BENCHMARKS)
EPISODE_COUNT = 1000
EPISODE_LEN = 1000
RLIABLE_REPS = 50000
XLABEL_Y_COORD_PER_TASK = -0.8
XLABEL_Y_COORD_AGGREGATED = -0.8

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

# === Specify W&B run IDs===
# TODO: Add MT-TD3-Latent-1hot, MT-TD3-Latent-Label, MT-TD3-Latent-FiLM, MT-TD3-Latent-Concat
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


# === Set up W&B ===
wandb.login()
wandb_api = wandb.Api()


# === Helpers ===
def save_fig(fig, name):
    fig.savefig(f"{name}.pdf", format="pdf", bbox_inches="tight")


def fetch_wandb_returns(benchmark: str, run_path):
    def column(*parts):
        return ".".join(parts)  # Build metric name

    # Get entire history for this run
    run = wandb_api.run(os.path.join(ENTITY_AND_PROJECT, run_path))
    history = run.history(samples=999_999_999)

    # Determine time step at (or immediately after) which to read the return from
    task_count = len(TASKS[benchmark])
    env_step_task = EPISODE_COUNT * EPISODE_LEN
    env_step_total = env_step_task * task_count

    # Get mean episodic return (normalized to range [0, 1])
    x = column(EVAL, ENV_STEP)
    y = column(EVAL, EPISODIC_RETURN_MEAN)
    total_return = history[history[x] >= env_step_total][y].iloc[0] / EPISODE_LEN

    # Get per-task episodic returns
    per_task_returns = {}
    for task_name in TASKS[benchmark]:
        x = column(EVAL, task_name, ENV_STEP)
        y = column(EVAL, task_name, EPISODIC_RETURN)
        per_task_returns[task_name] = history[history[x] >= env_step_task][y].iloc[0]
        per_task_returns[task_name] /= EPISODE_LEN  # Normalize to range [0, 1]

    return total_return, per_task_returns


def collect_returns(benchmark: str):
    total_returns = {}
    task_returns = {}

    for algo, run_paths in RUN_IDS[benchmark].items():
        print(f"{algo}@{benchmark}: Using {len(run_paths)} valid runs")
        totals = []
        per_tasks = {task: [] for task in TASKS[benchmark]}

        for run_path in run_paths:
            total, per_task = fetch_wandb_returns(benchmark, run_path)
            totals.append(total)
            for task, ret in per_task.items():
                per_tasks[task].append(ret)

        total_returns[algo] = np.array(totals).reshape(-1, 1)  # (num_runs, num_tasks=1)
        task_returns[algo] = {
            task: np.array(ret).reshape(-1, 1) for task, ret in per_tasks.items()
        }

    return total_returns, task_returns


# === Read normalized returns from W&B runs ===
normalized_mean_returns = {}
normalized_per_task_returns = {}
for benchmark in BENCHMARKS:
    print(f"Collect scores for '{benchmark}'...")
    mean_returns, per_task_returns = collect_returns(benchmark)
    normalized_mean_returns[benchmark] = mean_returns
    normalized_per_task_returns[benchmark] = per_task_returns


# === Plot algorithm comparison using all experiment data (across benchmarks) ===
aggregate_func_total = lambda x: np.array(
    [
        metrics.aggregate_iqm(x),
        metrics.aggregate_mean(x),
        metrics.aggregate_median(x),
        metrics.aggregate_optimality_gap(x, gamma=1),
    ]
)
metric_names_total = ["IQM ↑", "Mean ↑", "Median ↑", "Optimality Gap ↓"]

# For each algo, arrange scores as a matrix of shape (#seeds x #benchmarks) = (5, 3)
# NOTE: benchmark (e.g. walker-all) = rliable "task" (no task-specific scores required)
returns_per_algo = {}
for algo in algos:
    # Merge across benchmarks
    algo_returns = [
        normalized_mean_returns[benchmark][algo] for benchmark in BENCHMARKS
    ]
    returns_per_algo[algo] = np.concatenate(algo_returns, axis=1)
aggregate_scores, aggregate_cis = rly.get_interval_estimates(
    returns_per_algo, aggregate_func_total, reps=RLIABLE_REPS
)

fig, axes = plot_utils.plot_interval_estimates(
    aggregate_scores,
    aggregate_cis,
    metric_names=metric_names_total,
    algorithms=algos,
    xlabel_y_coordinate=XLABEL_Y_COORD_AGGREGATED,
    xlabel="",  # Normalized mean returns across 3 task sets: walker-all, spin-dog, and final-4",
    colors=COLORS,
)
save_fig(fig, os.path.join(OUT_DIR, EXPERIMENT, f"rliable_total"))


# === Compare algorithms in more detailed scope (per benchmark and per task) ===
for benchmark in BENCHMARKS:
    task_count = len(TASKS[benchmark])
    per_task_scores = {}
    per_task_cis = {}

    for task in TASKS[benchmark]:
        task_returns = {
            algo: normalized_per_task_returns[benchmark][algo][task] for algo in algos
        }
        aggregate_scores_task, aggregate_cis_task = rly.get_interval_estimates(
            task_returns, metrics.aggregate_iqm, reps=RLIABLE_REPS
        )

        if len(per_task_scores.keys()) == 0:
            # Initialize dict on first iteration
            for algo, score in aggregate_scores_task.items():
                per_task_scores[algo] = [score]
                per_task_cis[algo] = [
                    [aggregate_cis_task[algo][0]],
                    [aggregate_cis_task[algo][1]],
                ]
        else:
            for algo, score in aggregate_scores_task.items():
                per_task_scores[algo].append(score)
                per_task_cis[algo][0].append(aggregate_cis_task[algo][0])
                per_task_cis[algo][1].append(aggregate_cis_task[algo][1])

    # Add NaNs to fill up with white space later -> looks more consistent in the paper
    ghost_count = MAX_ALGO_COUNT - task_count
    for ghost in range(ghost_count):
        for algo, _ in aggregate_scores_task.items():
            per_task_scores[algo].append(np.nan)
            nan = np.empty_like(aggregate_cis_task[algo][0])
            nan[:] = np.nan
            per_task_cis[algo][0].append(nan)
            per_task_cis[algo][1].append(nan)

    for k in aggregate_scores.keys():
        per_task_scores[k] = np.array(per_task_scores[k])
        per_task_cis[k] = np.array(per_task_cis[k])

    # === Plot per-task IQM ===
    fig, axes = plot_utils.plot_interval_estimates(
        per_task_scores,
        per_task_cis,
        metric_names=[f"IQM {task} ↑" for task in TASKS[benchmark]]
        + [""] * ghost_count,
        algorithms=algos,
        xlabel="",
        # xlabel_y_coordinate=XLABEL_Y_COORD_PER_TASK,
        # xlabel=f"Normalized return per {benchmark} environment",  # Rather use caption
        colors=COLORS,
    )
    for i in range(task_count, MAX_ALGO_COUNT):
        axes[i].axis("off")  # Hide the empty subplot

    # Make all plots the same size for more consistency when displayed in the document
    fig_width = 13.6
    fig_height = 0.74
    fig.set_size_inches(fig_width, fig_height)

    # Avoid overlapping x labels
    for i, ax in enumerate(axes):
        ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=3))

    save_fig(fig, os.path.join(OUT_DIR, EXPERIMENT, f"rliable_per-task_{benchmark}"))


# === Compute PIs (probabilitis of improvement) ===
pi_key = "MT-TD3-Latent,MT-TD3-Direct"
xy = {pi_key: (returns_per_algo["MT-TD3-Latent"], returns_per_algo["MT-TD3-Direct"])}
pi_scores, pi_cis = rly.get_interval_estimates(
    xy, metrics.probability_of_improvement, reps=RLIABLE_REPS
)
# Take note of PI but don't plot it here
print("Probability of improvement:", pi_scores, pi_cis)


# === ChatGPT code to make per-task plots same size by padding, not stretching ===

# Option A
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas


def pad_figure(fig, target_size=(8, 6), outfile="padded.pdf"):
    # Render original to a buffer
    canvas = FigureCanvas(fig)
    fig_width, fig_height = fig.get_size_inches()
    buf = fig.canvas.buffer_rgba()

    # Create a new figure with target size
    new_fig = plt.figure(figsize=target_size)
    ax = new_fig.add_axes([0, 0, 1, 1])
    ax.axis("off")

    # Place the original figure as an image centered in the new figure
    ax.imshow(buf, extent=[0, fig_width, 0, fig_height])
    ax.set_xlim(0, target_size[0])
    ax.set_ylim(0, target_size[1])

    new_fig.savefig(outfile, bbox_inches="tight")
    plt.close(new_fig)


# Example usage
# fig = rliable.plot(...)
# pad_figure(fig, target_size=(8, 6), outfile="plot_padded.pdf")


# Option B
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
import matplotlib.image as mpimg


def save_with_padding(fig, target_size=(8, 6), outfile="padded.pdf"):
    # Save original figure to a temp file
    tmpfile = "_tmp_plot.pdf"
    fig.savefig(tmpfile, bbox_inches="tight")

    # Read it back in
    img = mpimg.imread(tmpfile)

    # Create bigger canvas
    new_fig = plt.figure(figsize=target_size)
    ax = new_fig.add_axes([0, 0, 1, 1])
    ax.axis("off")

    # Add image centered without stretching
    ab = AnnotationBbox(
        OffsetImage(img, zoom=1), (0.5, 0.5), frameon=False, box_alignment=(0.5, 0.5)
    )
    ax.add_artist(ab)

    # Save padded version
    new_fig.savefig(outfile, bbox_inches="tight", pad_inches=0)
    plt.close(new_fig)


# Example:
# fig = rliable.plot(...)
# save_with_padding(fig, target_size=(8, 6), outfile="plot_padded.pdf")
