#!/usr/bin/env python3
import os

# Use osmesa for headless video rendering (on GPU)
os.environ["MUJOCO_GL"] = "osmesa"
os.environ["PYOPENGL_PLATFORM"] = "osmesa"

from dataclasses import dataclass, field
from typing import Any, List, Optional

import hydra
from hydra.core.config_store import ConfigStore
from iqrl import iQRLConfig
from omegaconf import MISSING, OmegaConf
from utils import LUMIConfig, SlurmConfig


def envs_to_name(envs):
    return "_".join(f"{body}-{task}" for body, task in envs)


OmegaConf.register_new_resolver("envs_to_name", envs_to_name)


@dataclass
class TrainConfig:
    """Training config used in train.py"""

    defaults: List[Any] = field(
        default_factory=lambda: [
            "_self_",
            {"agent": "iqrl"},
            {"env": "dog-run"},  # envs are specified in cfgs/env/
            # Use submitit to launch slurm jobs on cluster w/ multirun
            {"override hydra/launcher": "slurm"},
            {"override hydra/job_logging": "colorlog"},  # Make logging colourful
            {"override hydra/hydra_logging": "colorlog"},  # Make logging colourful
        ]
    )

    # Configure envs as [body_name, task_name] items (overridden by defaults list)
    envs: list[list[str]] = MISSING
    name: str = MISSING

    # Agent (overridden by defaults list)
    agent: iQRLConfig = field(default_factory=iQRLConfig)

    # Experiment: General parameters
    max_episode_steps: int = 1000  # Max episode length
    num_episodes: int = 3000  # Number of training episodes per environment
    random_episodes: int = 10  # Number of random episodes at start
    action_repeat: int = 2
    buffer_size: int = 2_000_000  # Replay buffer size, per task
    prefetch: int = 5
    seed: int = 42
    checkpoint: Optional[str] = None  # /file/path/to/checkpoint
    device: str = "cuda"  # "cpu" or "cuda" etc
    verbose: bool = False  # if true print training progress

    # Experiment: Offline data
    use_offline_data: bool = False  # Train fully offline (but evaluate still online)
    normalize_states: bool = False  # Only takes effect if use_offline_data==True
    max_offline_episodes_per_task: int = 1000  # Limit offline episodes to reduce memory

    # Evaluation
    eval_only: bool = False  # Skip training (useful when loading checkpoint)
    eval_every_episodes: int = 20
    num_eval_episodes: int = 10
    capture_eval_video: bool = False  # Fails on AMD GPU so set to False
    log_per_task_sa: bool = False  # Log task-specific state & act ranges
    log_per_task_q: bool = False  # Log task-specific Q-values
    visualize_latent_states: bool = False  # Visualize latent state space using t-SNE
    visualize_latent_actions: bool = False  # Visualize latent action space using t-SNE
    visualize_body_embeddings: bool = False  # Visualize body embeddings using t-SNE
    visualize_task_embeddings: bool = False  # Visualize task embeddings using t-SNE
    verify_dyn_and_rew: bool = False  # Run dynamics for long and check rewards

    # W&B config
    use_wandb: bool = False
    wandb_project_name: str = "iqrl"
    run_name: str = "iqrl-${now:%Y-%m-%d_%H-%M-%S}"

    # Override the Hydra config to get better dir structure with W&B
    hydra: Any = field(
        default_factory=lambda: {
            "run": {"dir": "output/hydra/${hydra.job.name}/${now:%Y-%m-%d_%H-%M-%S}"},
            "verbose": False,
            "job": {"chdir": True},
            "sweep": {"dir": "${hydra.run.dir}", "subdir": "${hydra.job.num}"},
        }
    )


cs = ConfigStore.instance()
cs.store(name="train", node=TrainConfig)
cs.store(name="iqrl", group="agent", node=iQRLConfig)
cs.store(name="slurm", group="hydra/launcher", node=SlurmConfig)
cs.store(name="lumi", group="hydra/launcher", node=LUMIConfig)


@hydra.main(version_base="1.3", config_path="./cfgs", config_name="train")
def cluster_safe_train(cfg: TrainConfig):
    """Wrapper to ensure errors are logged properly when using hydra's submitit launcher

    This wrapper function is used to circumvent this bug in Hydra
    See https://github.com/facebookresearch/hydra/issues/2664
    """
    import sys
    import traceback

    try:
        train(cfg)
    except BaseException:
        traceback.print_exc(file=sys.stderr)
        raise
    finally:
        # flush everything
        sys.stdout.flush()
        sys.stderr.flush()


def train(cfg: TrainConfig):
    import logging
    import random
    import time
    from functools import partial

    from hydra.core.hydra_config import HydraConfig
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import seaborn as sns
    from sklearn.manifold import TSNE
    from termcolor import colored
    from tensordict import (
        LazyStackedTensorDict,
        TensorDict,
        pad_sequence,
        set_get_defaults_to_none,
    )
    from tensordict.nn import TensorDictModule
    from torchrl.data.tensor_specs import BoundedContinuous
    from torchrl.envs import ParallelEnv
    from torchrl.record.loggers.wandb import WandbLogger
    import torch
    import wandb

    from envs import make_env
    from iqrl import iQRL
    import utils.helper as h
    from utils import ReplayBuffer

    set_get_defaults_to_none(True)  # Useful for e.g. `obs.get(optional_param)`

    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger(__name__)

    assert not (cfg.agent.use_fsq and cfg.agent.use_simnorm), "Conflict: FSQ, SimNorm"
    assert cfg.agent.Q_and_rew_loss in ["mse", "soft-ce"], "Unsupported Q_and_rew_loss"
    assert cfg.agent.rl_algo in ["TD3", "SAC"], "Only TD3 and SAC are supported"
    assert cfg.agent.obs_types == ["state"], "Only obs_types=['state'] is supported"
    assert cfg.agent.enc_update_freq == 1, "enc_update_freq!=1 currently not supported"
    assert not cfg.verify_dyn_and_rew or cfg.agent.use_rew_loss, "Can't verify reward"
    if cfg.visualize_body_embeddings or cfg.visualize_task_embeddings:
        assert cfg.agent.context_dim is not None, "No embeddings found to visualize"

    ###### Fix seed for reproducibility ######
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.backends.cudnn.deterministic = True

    cfg.device = (
        "cuda" if torch.cuda.is_available() and (cfg.device == "cuda") else "cpu"
    )

    ###### Initialise W&B ######
    env_names = [f"{body_name}-{task_name}" for body_name, task_name in cfg.envs]
    env_count = len(env_names)
    writer = WandbLogger(
        exp_name=cfg.run_name,
        offline=not cfg.use_wandb,
        project=cfg.wandb_project_name,
        # group=f"{cfg.env_name}-{cfg.task_name}", TODO: what are groups in multi-task?
        tags=env_names + [f"seed={cfg.seed}"],
        save_code=True,
    )
    writer.log_hparams(cfg)
    writer.log_hparams(
        {"hydra": OmegaConf.to_container(HydraConfig.get(), throw_on_missing=False)}
    )

    ###### Setup environment for training/evaluation/video recording ######
    body_names = [body_name for body_name, _ in cfg.envs]
    task_names = [task_name for _, task_name in cfg.envs]
    body_str_to_id = h.seq_to_id(body_names)
    task_str_to_id = h.seq_to_id(task_names)
    n_body = len(set(body_names))
    n_task = len(set(task_names))

    common_kwargs_for_make_env = {
        "seed": cfg.seed,
        "frame_skip": cfg.action_repeat,
        "from_pixels": False,
        "pixels_only": False,
        "logger": writer,
        "n_body": n_body,
        "n_task": n_task,
        "device": cfg.device,
    }
    create_fn = [
        partial(
            make_env,
            env_name=body_name,
            task_name=task_name,
            body_id=torch.tensor([body_str_to_id[body_name]], device=cfg.device),
            task_id=torch.tensor([task_str_to_id[task_name]], device=cfg.device),
            record_video=False,  # No need, video_envs will record videos
            **common_kwargs_for_make_env,
        )
        for body_name, task_name in cfg.envs
    ]

    obs_specs = []
    act_specs = []
    for fn in create_fn:
        subenv_dummy = fn()
        obs_specs.append(subenv_dummy.observation_spec["observation"])
        act_specs.append(subenv_dummy.action_spec)
        assert isinstance(
            subenv_dummy.action_spec, BoundedContinuous
        ), "Only continuous actions supported"
        subenv_dummy.close()

    od = [obs_spec["state"].shape[0] for obs_spec in obs_specs]
    ad = [act_spec.shape[0] for act_spec in act_specs]
    create_fn = [
        partial(fn, obs_dim=od[i], act_dim=ad[i], max_act_dim=max(ad))
        for i, fn in enumerate(create_fn)
    ]

    env = ParallelEnv(
        env_count,
        [partial(fn, use_offline_data=cfg.agent.use_offline_data) for fn in create_fn],
    )
    eval_env = ParallelEnv(
        env_count,
        [partial(fn, use_offline_data=False) for fn in create_fn],
    )
    if cfg.capture_eval_video:
        video_envs = [
            make_env(
                env_name=body_name,
                task_name=task_name,
                body_id=torch.tensor([body_str_to_id[body_name]], device=cfg.device),
                task_id=torch.tensor([task_str_to_id[task_name]], device=cfg.device),
                record_video=cfg.capture_eval_video,
                use_offline_data=False,
                obs_dim=od[i],
                act_dim=ad[i],
                max_act_dim=max(ad),
                **common_kwargs_for_make_env,
            )
            for i, (body_name, task_name) in enumerate(cfg.envs)
        ]

    ###### Prepare replay buffer ######
    nstep = max(cfg.agent.get("nstep", 1), cfg.agent.get("horizon", 1))
    if cfg.agent.batch_size % env_count != 0:
        cfg.agent.batch_size -= cfg.agent.batch_size % env_count
        logger.info(f"Using batch_size={cfg.agent.batch_size} for even sampling")
    rb = ReplayBuffer(
        buffer_size=cfg.buffer_size,
        batch_size=cfg.agent.batch_size,
        buffer_count=env_count,
        nstep=nstep,
        gamma=cfg.agent.gamma,
        prefetch=cfg.prefetch,
        pin_memory=True,  # will be set to False if device=="cpu"
        device=cfg.device,
    )

    ###### Init agent ######
    ids_to_dims = {}  # (body ID, task ID) -> (obs dim, action dim), all integers
    for i in range(env_count):
        body_id = np.argmax(body_str_to_id[cfg.envs[i][0]]).item()
        task_id = np.argmax(task_str_to_id[cfg.envs[i][1]]).item()
        o = np.array(obs_specs[i]["state"].shape).prod().item()
        a = np.array(act_specs[i].shape).prod().item()
        ids_to_dims[(body_id, task_id)] = (o, a)
    agent = iQRL(
        cfg=cfg.agent,
        obs_specs=obs_specs,
        act_specs=act_specs,
        n_body=n_body,
        n_task=n_task,
        ids_to_dims=ids_to_dims,
    )
    # Load state dict into this agent from filepath (or dictionary)
    if cfg.checkpoint is not None:
        state_dict = torch.load(cfg.checkpoint, map_location=cfg.device)
        agent.load_state_dict(state_dict["model"])
        logger.info(f"Loaded checkpoint from {cfg.checkpoint}")
        if cfg.random_episodes != 0:
            cfg.random_episodes = 0
            logger.info("Set random_episodes=0 because checkpoint was loaded")

    policy_module = TensorDictModule(
        lambda obs: agent.select_action(obs, eval_mode=False),
        in_keys=["observation"],
        out_keys=["action"],
    )
    eval_policy_module = TensorDictModule(
        lambda obs: agent.select_action(obs, eval_mode=True),
        in_keys=["observation"],
        out_keys=["action"],
    )

    ##### Print information about run #####
    mstep = (cfg.num_episodes * cfg.max_episode_steps) / 1e6
    total_params = int(agent.total_params / 1e6)
    writer.log_hparams({"total_params": agent.total_params})
    print(colored("Envs:", "yellow", attrs=["bold"]), "_".join(env_names))
    print(colored("Number of episodes:", "yellow", attrs=["bold"]), cfg.num_episodes)
    print(colored("Max number of env. steps:", "yellow", attrs=["bold"]), mstep, "M")
    print(colored("Action repeat:", "green", attrs=["bold"]), cfg.action_repeat)
    print(colored("Device:", "green", attrs=["bold"]), cfg.device)
    print(colored("Learnable parameters:", "green", attrs=["bold"]), f"{total_params}M")
    print(colored("Architecture:", "green", attrs=["bold"]), agent)

    def evaluate(
        cfg: TrainConfig, steps: list[int], episode_idx: int, start_time: float
    ) -> dict:
        """Evaluate agent in eval_env and log metrics"""
        eval_metrics = {env_name: {} for env_name in env_names}
        eval_start_time = time.time()
        with torch.no_grad():
            episodic_returns = {env_name: [] for env_name in env_names}
            if cfg.log_per_task_sa:
                all_states = {env_names[i]: [] for i in range(env_count)}
                all_actions = {env_names[i]: [] for i in range(env_count)}
            if cfg.verify_dyn_and_rew:
                eval_datas = []

            for _ in range(cfg.num_eval_episodes):
                eval_data = eval_env.rollout(
                    max_steps=cfg.max_episode_steps // cfg.action_repeat,
                    policy=eval_policy_module,
                    break_when_any_done=False,
                )

                for task_i, env_name in enumerate(env_names):
                    episodic_returns[env_name].append(
                        eval_data["next"]["episode_reward"][task_i][-1].cpu().item()
                    )

                if cfg.log_per_task_sa:
                    if isinstance(eval_data, TensorDict):
                        states = eval_data["observation"]["state"]
                        actions = eval_data["action"]
                    else:
                        states = eval_data["observation"].get_nestedtensor("state")
                        actions = eval_data.get_nestedtensor("action")
                    for task_i, env_name in enumerate(env_names):
                        all_states[env_name].append(states[task_i].cpu())
                        all_actions[env_name].append(actions[task_i].cpu())
                if cfg.verify_dyn_and_rew:
                    eval_datas.append(eval_data)

            for task_i, env_name in enumerate(env_names):
                ep_return = sum(episodic_returns[env_name]) / cfg.num_eval_episodes
                eval_metrics[env_name]["episodic_return"] = ep_return
                eval_metrics[env_name]["env_step"] = steps[task_i] * cfg.action_repeat
            eval_episodic_return_mean = np.mean(
                [eval_metrics[env_name]["episodic_return"] for env_name in env_names]
            )

        ##### Task-specific training metrics #####
        if cfg.log_per_task_sa or cfg.log_per_task_q:
            for task_i in range(env_count):
                task_metrics = agent.update(
                    replay_buffer=rb, num_new_transitions=500, fake=True, rb_idx=task_i
                )
                task_metrics["env_step"] = steps[task_i] * cfg.action_repeat

                if cfg.log_per_task_sa:
                    task_states = np.array(all_states[env_names[task_i]])
                    for dim in range(task_states.shape[-1]):
                        states_single = task_states[..., dim]
                        task_metrics.update(
                            {
                                f"state_min_{dim=}": states_single.min().item(),
                                f"state_max_{dim=}": states_single.max().item(),
                                f"state_mean_{dim=}": states_single.mean().item(),
                                f"state_std_{dim=}": states_single.std().item(),
                            }
                        )
                    task_actions = np.array(all_actions[env_names[task_i]])
                    for dim in range(task_actions.shape[-1]):
                        actions_single = task_actions[..., dim]
                        task_metrics.update(
                            {
                                f"action_min_{dim=}": actions_single.min().item(),
                                f"action_max_{dim=}": actions_single.max().item(),
                                f"action_mean_{dim=}": actions_single.mean().item(),
                                f"action_std_{dim=}": actions_single.std().item(),
                            }
                        )

                writer.log_scalar(name=f"{env_names[task_i]}/", value=task_metrics)

        ##### Overall eval metrics #####
        eval_metrics.update(
            {
                "episodic_return_mean": eval_episodic_return_mean,
                "elapsed_time": time.time() - start_time,
                "SPS": int(sum(steps) / (time.time() - start_time)),
                "episode_time": (time.time() - eval_start_time) / cfg.num_eval_episodes,
                "env_step": sum(steps) * cfg.action_repeat,
                "step": sum(steps),
                "episode": episode_idx,
            }
        )

        if cfg.verbose:
            logger.info(
                f"Episode {episode_idx} | Env Step {sum(steps)*cfg.action_repeat} | "
                f"Eval return (mean over envs) {eval_episodic_return_mean:.2f}"
            )

        ##### If desired, log videos and plots at beginning, midpoint and end of training #####
        next_eval_idx = episode_idx + cfg.eval_every_episodes
        is_first = episode_idx == 0
        is_middle = episode_idx <= cfg.num_episodes // 2 < next_eval_idx
        is_last = episode_idx == cfg.num_episodes  # Corresponds to final eval call
        if is_first or is_middle or is_last:
            if cfg.capture_eval_video:
                with torch.no_grad():
                    for video_env in video_envs:
                        video_env.rollout(
                            max_steps=cfg.max_episode_steps // cfg.action_repeat,
                            policy=eval_policy_module,
                            break_when_any_done=False,
                        )
                        video_env.transform.dump()

            if (
                cfg.visualize_latent_states
                or cfg.visualize_latent_actions
                or cfg.visualize_body_embeddings
                or cfg.visualize_task_embeddings
            ):

                def log_tsne(latent_data: np.ndarray, idx_name: str, val_name: str):
                    if latent_data.shape[0] == 1:
                        print("Can't run t-SNE with a single sample")
                        return
                    perp = min(30.0, latent_data.shape[0] - 1)
                    tsne = TSNE(verbose=1, max_iter=5000, perplexity=perp)
                    tsne_results = tsne.fit_transform(latent_data)
                    if idx_name == "Env":
                        idx = [env_names[i] for i in range(env_count) for _ in range(n)]
                    elif idx_name == "Embodiment":
                        idx = []
                        for b in body_names:
                            if b not in idx:
                                idx.append(b)
                    elif idx_name == "Task":
                        idx = []
                        for t in task_names:
                            if t not in idx:
                                idx.append(t)
                    tsne_data = pd.DataFrame(
                        {
                            idx_name: idx,
                            "t-SNE dim 1": tsne_results[:, 0],
                            "t-SNE dim 2": tsne_results[:, 1],
                        }
                    )

                    plt.figure(figsize=(16, 10))
                    plt.title(f"{val_name} ({episode_idx} episodes)")
                    tsne_plot = sns.scatterplot(
                        x="t-SNE dim 1",
                        y="t-SNE dim 2",
                        hue=idx_name,
                        palette=sns.color_palette("husl", len(set(idx))),
                        data=tsne_data,
                        legend="full",
                        alpha=0.7 if idx_name == "Env" else 1,
                    )
                    wandb.log(
                        {
                            f"tsne_latent_{val_name.replace(' ','_').lower()}": wandb.Image(
                                tsne_plot.get_figure()
                            )
                        }
                    )

                data = pad_sequence(eval_data, pad_dim=-1)  # Pad latest eval iter
                n = data.shape[1]  # Samples per env

                # t-SNE expects (n_samples, n_features) -> (env_count*n, latent_dim)
                with torch.no_grad():
                    if cfg.visualize_latent_states:
                        latent_states = agent.encoder.encode_obs(data["observation"])[
                            "state"
                        ]
                        latent_states = latent_states.flatten(0, 1).cpu().numpy()
                        log_tsne(latent_states, "Env", "Latent states")
                    if cfg.visualize_latent_actions:
                        latent_actions = agent.encoder.encode_action(
                            action=data["action"].to(cfg.device),
                            ctx=agent.encoder.get_context(data["observation"]),
                        )
                        latent_actions = latent_actions.flatten(0, 1).cpu().numpy()
                        log_tsne(latent_actions, "Env", "Latent actions")
                    if cfg.visualize_task_embeddings:
                        task_ids = torch.arange(n_task).long().to(cfg.device)
                        task_emb = agent.encoder._task_emb(task_ids).cpu().numpy()
                        log_tsne(task_emb, "Task", "Task embeddings")
                    if cfg.visualize_task_embeddings:
                        body_ids = torch.arange(n_body).long().to(cfg.device)
                        body_emb = agent.encoder._body_emb(body_ids).cpu().numpy()
                        log_tsne(body_emb, "Embodiment", "Body embeddings")

        ##### If desired, log how well the dynamics and reward model are working #####
        if cfg.verify_dyn_and_rew:
            max_t = cfg.max_episode_steps // cfg.action_repeat
            rew_mses = np.empty((env_count, max_t, cfg.num_eval_episodes))
            true_rews = np.empty((env_count, max_t, cfg.num_eval_episodes))

            for i, data in enumerate(eval_datas):
                rew_target = data["next"]["reward"]  # Shape: (env count, max_t, 1)
                true_rews[:, :, i] = rew_target.view((env_count, max_t))

                # Extract actions with shape (env count, max_t, max_act_dim)
                if isinstance(data, TensorDict):
                    actions = data["action"]
                else:
                    actions = data.get_nestedtensor("action")
                    actions = torch.nested.to_padded_tensor(actions, padding=0.0)

                # Encode initial env observation; z.shape is (env_count, L)
                z = agent.encoder.encode_obs(data["observation"][:, 0])["codes"]

                # Predict next max_t latent states and calculate reward diff on the way
                ctx_t = agent.encoder.get_context(data["observation"][..., 0])
                for t in range(max_t):
                    rew_pred_t = agent.encoder.reward(z, actions[:, t], ctx_t).detach()
                    if cfg.agent.Q_and_rew_loss == "soft-ce":
                        rew_pred_t = h.two_hot_inv(rew_pred_t, cfg.agent)
                    rew_mses[:, t, i] = ((rew_pred_t - rew_target[:, t]) ** 2).squeeze()

                    # Use dynamics model to move on to next latent observation
                    z = agent.encoder.trans(z, actions[:, t], ctx_t)["codes"]

            true_rews = true_rews.mean(axis=-1)
            rew_mses = rew_mses.mean(axis=-1)  # New shape: (env_count, max_t)

            # Plot result (1 line per env)
            df = pd.DataFrame(rew_mses)
            df["env"] = df.index  # add an environment label
            df_long = df.melt(id_vars="env", var_name="time", value_name="value")
            df_long["time"] = df_long["time"].astype(int)
            df_long["env_name"] = df_long["env"].map(lambda i: env_names[i])

            if cfg.checkpoint is None:
                run_id = "No checkpoint"
            else:
                run_id = cfg.checkpoint.split("/")[-3]  # YYYY-MM-DD

            plt.figure(figsize=(10, 6))
            sns.lineplot(
                data=df_long, x="time", y="value", hue="env_name", palette="tab10"
            )
            plt.xlabel("Env step")
            plt.ylabel("(rew_true - rew_pred)**2")
            plt.title(
                f"{run_id}: Reward discrepancy (avg. over {cfg.num_eval_episodes} eps.)"
            )
            plt.legend(title="Env")
            plt.tight_layout()

            plt.savefig("dyn_and_rew_check.pdf", format="pdf")
            plt.close()

            # Also plot true rewards
            df = pd.DataFrame(true_rews)
            df["env"] = df.index  # add an environment label
            df_long = df.melt(id_vars="env", var_name="time", value_name="value")
            df_long["time"] = df_long["time"].astype(int)
            df_long["env_name"] = df_long["env"].map(lambda i: env_names[i])
            plt.figure(figsize=(10, 6))
            sns.lineplot(
                data=df_long, x="time", y="value", hue="env_name", palette="tab10"
            )
            plt.xlabel("Env step")
            plt.ylabel("rew_true")
            plt.title(
                f"{run_id}: true rewards (avg. over {cfg.num_eval_episodes} eps.)"
            )
            plt.legend(title="Env")
            plt.tight_layout()

            plt.savefig("true_rews.pdf", format="pdf")
            plt.close()

        ##### Log rank of latent and active codebook percent #####
        batch = rb.sample(batch_size=agent.encoder.cfg.latent_dim)
        eval_metrics.update(agent.metrics(batch))

        ##### Log metrics to W&B or csv #####
        writer.log_scalar(name="eval/", value=eval_metrics)
        return eval_metrics

    if cfg.use_offline_data:
        print("Loading offline data into replay buffer...")

        rollout_blueprint = env.rollout(
            max_steps=cfg.max_episode_steps // cfg.action_repeat,
            policy=policy_module,
            break_when_any_done=True,  # Same-length episodes -> break when done
        )

        offline_data = []
        state_normalization = {}  # Needed during eval if cfg.normalize_states==True
        for i in range(env_count):
            MT30_DATA_DIR = os.path.join(
                os.environ.get("WRKDIR"), "data", "mt30", "per-task"
            )
            file_path = os.path.join(os.path.join(MT30_DATA_DIR, f"{env_names[i]}.pt"))
            task_data_raw = torch.load(file_path, weights_only=False).to(cfg.device)
            assert sorted(task_data_raw.keys()) == ["action", "obs", "reward"]

            task_data_raw = task_data_raw[: cfg.max_offline_episodes_per_task]

            task_blueprint = rollout_blueprint[i]
            new_shape = [task_data_raw.shape[0]] + list(task_blueprint.shape)

            task_data = task_blueprint.unsqueeze(0).expand(new_shape)
            task_data["observation"]["state"] = task_data_raw["obs"][:, :-1]
            task_data["next"]["observation"]["state"] = task_data_raw["obs"][:, 1:]
            task_data["action"] = task_data_raw["action"][:, 1:]
            task_data["reward"] = task_data_raw["reward"][:, 1:]
            # "done" and ("next", "terminated") remain False all the way, it's fine

            if cfg.normalize_states:
                # Normalize over all states of this specific body&task combination
                # States shape is (episode_count, ep_length, max_state_dim) -> dim=(0, 1)
                mean = task_data["observation"]["state"].mean(dim=(0, 1))
                std = task_data["observation"]["state"].std(dim=(0, 1)) + 1e-3
                task_data["observation"]["state"] = (
                    task_data["observation"]["state"] - mean
                ) / std
                task_data["next"]["observation"]["state"] = (
                    task_data["next"]["observation"]["state"] - mean
                ) / std

                body_id = np.argmax(body_str_to_id[cfg.envs[i][0]]).item()
                task_id = np.argmax(task_str_to_id[cfg.envs[i][1]]).item()
                state_normalization[(body_id, task_id)] = (mean, std)

            offline_data.append(task_data)

        if cfg.normalize_states:
            agent.set_state_norm(state_normalization)

        data = LazyStackedTensorDict.lazy_stack(offline_data, dim=0)
        data = pad_sequence(data, pad_dim=-1)  # LazyStackedTensorDict -> TensorDict
        data = data.flatten(0, 1)  # Merge first 2 dims: env_count and episode_count

        rb.extend(data)

    steps = [0 for _ in range(env_count)]  # Some envs might step more than others
    start_time = time.time()
    for episode_idx in range(cfg.num_episodes):
        if not cfg.use_offline_data:
            ##### Rollout the policy in the environment #####
            with torch.no_grad():
                data = env.rollout(
                    max_steps=cfg.max_episode_steps // cfg.action_repeat,
                    policy=policy_module,
                    break_when_any_done=False,
                )
            ##### Add data to the replay buffer #####
            data = pad_sequence(data, pad_dim=-1)  # LazyStackedTensorDict -> TensorDict
            rb.extend(data)

        if cfg.eval_only:
            break  # Only use final eval at the very end of this function

        if episode_idx == 0:
            if not cfg.use_offline_data:
                print(colored("First episodes data:", "green", attrs=["bold"]), data)

            # Evaluate the initial agent
            _ = evaluate(
                cfg, steps=steps, episode_idx=episode_idx, start_time=start_time
            )

        if not cfg.use_offline_data:
            ##### Log episode metrics #####
            num_new_transitions = 0
            for i in range(env_count):
                step_count_i = data["next"]["step_count"][i][-1].cpu().sum().item()
                steps[i] += step_count_i
                num_new_transitions += step_count_i

            episode_rewards = [
                data["next"]["episode_reward"][i][-1].cpu().item()
                for i in range(env_count)
            ]

            episodic_return_mean = sum(episode_rewards) / env_count
            if cfg.verbose:
                logger.info(
                    f"Episode {episode_idx} | "
                    f"Env Step {sum(steps)*cfg.action_repeat} | "
                    f"Train return (mean over envs) {episodic_return_mean:.2f} | "
                    f"Train return per env {' '.join(map(str, episode_rewards))}"
                )
            rollout_metrics = {
                "episodic_return_mean": episodic_return_mean,
                "episodic_length": num_new_transitions // env_count,
                "env_step": sum(steps) * cfg.action_repeat,
            }
            rollout_metrics.update({env_name: {} for env_name in env_names})
            for i in range(env_count):
                rollout_metrics[env_names[i]]["episodic_return"] = episode_rewards[i]

            writer.log_scalar(name="rollout/", value=rollout_metrics)
        else:
            for i in range(env_count):
                steps[i] += 1  # Avoid logging more than once per step

        ##### Train agent (after collecting some random episodes) #####
        if cfg.use_offline_data or episode_idx > cfg.random_episodes - 1:
            update_start_time = time.time()
            train_metrics = agent.update(
                replay_buffer=rb, num_new_transitions=num_new_transitions
            )
            train_metrics["update_time"] = time.time() - update_start_time
            train_metrics["env_step"] = sum(steps) * cfg.action_repeat
            writer.log_scalar(name="train/", value=train_metrics)
            torch.save({"model": agent.state_dict()}, "./checkpoint")
            if episode_idx % cfg.eval_every_episodes == 0:
                evaluate(
                    cfg,
                    steps=steps,
                    episode_idx=episode_idx,
                    start_time=start_time,
                )

        # Release some GPU memory (if possible)
        torch.cuda.empty_cache()

    # Evaluate the final agent
    _ = evaluate(cfg, steps=steps, episode_idx=cfg.num_episodes, start_time=start_time)

    env.close()
    eval_env.close()


if __name__ == "__main__":
    cluster_safe_train()  # pyright: ignore
