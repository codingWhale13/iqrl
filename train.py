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
    num_episodes: int = 1000  # Number of training episodes per environment
    random_episodes: int = 10  # Number of random episodes at start
    action_repeat: int = 2
    buffer_size: int = 1_000_000  # Replay buffer size, per task
    prefetch: int = 5
    seed: int = 42
    checkpoint: Optional[str] = None  # /file/path/to/checkpoint
    device: str = "cuda"  # "cpu" or "cuda" etc
    verbose: bool = False  # if true print training progress

    # Evaluation
    eval_every_episodes: int = 20
    num_eval_episodes: int = 10
    capture_eval_video: bool = False  # Fails on AMD GPU so set to False
    log_per_task_q: bool = False  # Log task-specific Q-values

    # W&B config
    use_wandb: bool = False
    wandb_project_name: str = "iqrl"
    run_name: str = "iqrl-${now:%Y-%m-%d_%H-%M-%S}"

    # Override the Hydra config to get better dir structure with W&B
    hydra: Any = field(
        default_factory=lambda: {
            "run": {
                "dir": "/scratch/work/kielen1/experiments/iqrl/output/hydra/${hydra.job.name}/${now:%Y-%m-%d_%H-%M-%S}"
            },
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
    import numpy as np
    from termcolor import colored
    from tensordict import pad_sequence, set_get_defaults_to_none
    from tensordict.nn import TensorDictModule
    from torchrl.data.tensor_specs import BoundedContinuous
    from torchrl.envs import ParallelEnv
    from torchrl.record.loggers.wandb import WandbLogger
    import torch

    from envs import make_env
    from iqrl import iQRL
    import utils.helper as h
    from utils import ReplayBuffer

    set_get_defaults_to_none(True)  # Useful for e.g. `obs.get(optional_param)`

    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger(__name__)

    assert cfg.agent.condition_layer in ["first", "all"], "Unknown condition_layer"
    assert cfg.agent.Q_and_rew_loss in ["mse", "soft-ce"], "Unsupported Q_and_rew_loss"
    assert cfg.agent.obs_types == ["state"], "Only obs_types=['state'] is supported"
    assert cfg.agent.enc_update_freq == 1, "enc_update_freq!=1 currently not supported"
    if not cfg.agent.use_representation_learning:
        logger.info("No representation learning => encoder and FSQ will not be used")
        cfg.agent.use_obs_encoder = False
        cfg.agent.use_fsq = False

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
    body_names_org = [body_name for body_name, _ in cfg.envs]  # Keep these for make_env
    task_names_org = [task_name for _, task_name in cfg.envs]  # Keep these for make_env
    body_names = [body_name for body_name, _ in cfg.envs]
    task_names = [task_name for _, task_name in cfg.envs]
    body_str_to_id = h.seq_to_id(body_names)
    task_str_to_id = h.seq_to_id(task_names)
    n_body = len(set(body_names))
    n_task = len(set(task_names))
    print(f"{n_body=}", f"{n_task=}")
    print(f"{body_names=}")
    print(f"{task_names=}")
    print(f"{body_str_to_id=}")
    print(f"{task_str_to_id=}")

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
            env_name=body_names_org[i],
            task_name=task_names_org[i],
            body_id=torch.tensor([body_str_to_id[body_names[i]]], device=cfg.device),
            task_id=torch.tensor([task_str_to_id[task_names[i]]], device=cfg.device),
            record_video=False,  # No need, video_envs will record videos
            **common_kwargs_for_make_env,
        )
        for i in range(env_count)
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

    ad = [act_spec.shape[0] for act_spec in act_specs]
    create_fn = [
        partial(fn, act_dim=ad[i], max_act_dim=max(ad))
        for i, fn in enumerate(create_fn)
    ]

    env = ParallelEnv(env_count, create_fn)
    eval_env = ParallelEnv(env_count, create_fn)
    if cfg.capture_eval_video:
        video_envs = [
            make_env(
                env_name=body_names_org[i],
                task_name=task_names_org[i],
                body_id=torch.tensor(
                    [body_str_to_id[body_names[i]]], device=cfg.device
                ),
                task_id=torch.tensor(
                    [task_str_to_id[task_names[i]]], device=cfg.device
                ),
                record_video=cfg.capture_eval_video,
                act_dim=ad[i],
                max_act_dim=max(ad),
                **common_kwargs_for_make_env,
            )
            for i in range(env_count)
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
        body_id = body_str_to_id[body_names[i]]
        task_id = task_str_to_id[task_names[i]]
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

            for task_i, env_name in enumerate(env_names):
                ep_return = sum(episodic_returns[env_name]) / cfg.num_eval_episodes
                eval_metrics[env_name]["episodic_return"] = ep_return
                eval_metrics[env_name]["env_step"] = steps[task_i] * cfg.action_repeat
            eval_episodic_return_mean = np.mean(
                [eval_metrics[env_name]["episodic_return"] for env_name in env_names]
            )

        ##### Task-specific training metrics #####
        if cfg.log_per_task_q:
            for task_i in range(env_count):
                task_metrics = agent.update(
                    replay_buffer=rb, num_new_transitions=500, fake=True, rb_idx=task_i
                )
                task_metrics["env_step"] = steps[task_i] * cfg.action_repeat
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

        ##### Log rank of latent and active codebook percent #####
        if cfg.agent.use_representation_learning:
            batch = rb.sample(batch_size=agent.encoder.cfg.latent_dim)
            eval_metrics.update(agent.metrics(batch))

        ##### Log metrics to W&B or csv #####
        writer.log_scalar(name="eval/", value=eval_metrics)
        return eval_metrics

    steps = [0 for _ in range(env_count)]  # Some envs might step more than others
    start_time = time.time()
    for episode_idx in range(cfg.num_episodes):
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

        if episode_idx == 0:
            print(colored("First episodes data:", "green", attrs=["bold"]), data)

            # Evaluate the initial agent
            _ = evaluate(
                cfg, steps=steps, episode_idx=episode_idx, start_time=start_time
            )

        ##### Log episode metrics #####
        num_new_transitions = 0
        for i in range(env_count):
            step_count_i = data["next"]["step_count"][i][-1].cpu().sum().item()
            steps[i] += step_count_i
            num_new_transitions += step_count_i

        episode_rewards = [
            data["next"]["episode_reward"][i][-1].cpu().item() for i in range(env_count)
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

        ##### Train agent (after collecting some random episodes) #####
        if episode_idx > cfg.random_episodes - 1:
            update_start_time = time.time()
            train_metrics = agent.update(
                replay_buffer=rb, num_new_transitions=num_new_transitions
            )
            train_metrics["update_time"] = time.time() - update_start_time
            train_metrics["env_step"] = sum(steps) * cfg.action_repeat
            writer.log_scalar(name="train/", value=train_metrics)
            if episode_idx % cfg.eval_every_episodes == 0:
                if cfg.verbose:
                    logger.info(f"Saving model checkpoint for episode {episode_idx}")
                torch.save({"model": agent.state_dict()}, "./checkpoint")
                evaluate(
                    cfg,
                    steps=steps,
                    episode_idx=episode_idx,
                    start_time=start_time,
                )

        # Release some GPU memory (if possible)
        torch.cuda.empty_cache()

    # Save final checkpoint and evaluate the final agent
    if cfg.verbose:
        logger.info("Saving final model checkpoint")
    torch.save({"model": agent.state_dict()}, "./checkpoint")
    _ = evaluate(cfg, steps=steps, episode_idx=cfg.num_episodes, start_time=start_time)

    env.close()
    eval_env.close()


if __name__ == "__main__":
    cluster_safe_train()  # pyright: ignore
