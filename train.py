#!/usr/bin/env python3
import os

os.environ["MUJOCO_GL"] = "osmesa"  # Needed for video recording on GPU

from dataclasses import dataclass, field
from functools import partial
from typing import Any, List, Optional

import hydra
from hydra.core.config_store import ConfigStore
from iqrl import iQRLConfig
from omegaconf import MISSING, OmegaConf
from torchrl.envs import SerialEnv
from utils import LUMIConfig, SlurmConfig
import utils.helper as h


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

    # Experiment
    max_episode_steps: int = 1000  # Max episode length
    num_episodes: int = 3000  # Number of training episodes (3M env steps)
    random_episodes: int = 10  # Number of random episodes at start
    action_repeat: int = 2
    buffer_size: int = 10_000_000
    prefetch: int = 5
    seed: int = 42
    checkpoint: Optional[str] = None  # /file/path/to/checkpoint
    device: str = "cuda"  # "cpu" or "cuda" etc
    verbose: bool = False  # if true print training progress

    # Evaluation
    eval_every_episodes: int = 20
    num_eval_episodes: int = 10
    capture_eval_video: bool = False  # Fails on AMD GPU so set to False
    log_dormant_neuron_ratio: bool = False

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

    import numpy as np
    import torch
    from envs import make_env
    from iqrl import iQRL
    from tensordict.nn import TensorDictModule
    from termcolor import colored
    from torchrl.data.tensor_specs import BoundedTensorSpec
    from torchrl.record.loggers.wandb import WandbLogger
    from utils import ReplayBuffer

    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger(__name__)

    assert cfg.agent.obs_types == ["state"], "only obs_types=['state'] is supported"

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

    ###### Setup environment for training/evaluation/video recording ######
    body_str_to_id = h.seq_to_1hot([body_name for body_name, _ in cfg.envs])
    task_str_to_id = h.seq_to_1hot([task_name for _, task_name in cfg.envs])

    common_kwargs_for_make_env = {
        "seed": cfg.seed,
        "frame_skip": cfg.action_repeat,
        "from_pixels": False,
        "pixels_only": False,
        "logger": writer,
    }
    create_env_fn = [
        partial(
            make_env,
            env_name=body_name,
            task_name=task_name,
            body_id=body_str_to_id[body_name],
            task_id=task_str_to_id[task_name],
            record_video=False,
            **common_kwargs_for_make_env,
        )
        for body_name, task_name in cfg.envs
    ]
    env = SerialEnv(env_count, create_env_fn)
    eval_env = SerialEnv(env_count, create_env_fn)
    video_envs = [
        make_env(
            env_name=body_name,
            task_name=task_name,
            body_id=body_str_to_id[body_name],
            task_id=task_str_to_id[task_name],
            record_video=cfg.capture_eval_video,
            **common_kwargs_for_make_env,
        )
        for body_name, task_name in cfg.envs
    ]

    assert isinstance(
        env.action_spec, BoundedTensorSpec
    ), "only continuous action space is supported"

    ###### Prepare replay buffer ######
    nstep = max(cfg.agent.get("nstep", 1), cfg.agent.get("horizon", 1))
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
    # iQRL components should not worry about batch dimensions
    subenv_dummy = make_env(
        env_name=cfg.envs[0][0],
        task_name=cfg.envs[0][1],
        body_id=body_str_to_id[cfg.envs[0][0]],
        task_id=task_str_to_id[cfg.envs[0][1]],
        record_video=False,
        **common_kwargs_for_make_env,
    )
    agent = iQRL(
        cfg=cfg.agent,
        obs_spec=subenv_dummy.observation_spec["observation"],
        act_spec=subenv_dummy.action_spec,
    )
    # Load state dict into this agent from filepath (or dictionary)
    if cfg.checkpoint is not None:
        state_dict = torch.load(cfg.checkpoint)
        agent.load_state_dict(state_dict["model"])
        logger.info(f"Loaded checkpoint from {cfg.checkpoint}")

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
    steps = (cfg.num_episodes * cfg.max_episode_steps) / 1e6
    total_params = int(agent.total_params / 1e6)
    writer.log_hparams({"total_params": agent.total_params})
    print(colored("Envs:", "yellow", attrs=["bold"]), "_".join(env_names))
    print(colored("Number of episodes:", "yellow", attrs=["bold"]), cfg.num_episodes)
    print(colored("Max number of env. steps:", "yellow", attrs=["bold"]), steps, "M")
    print(colored("Action repeat:", "green", attrs=["bold"]), cfg.action_repeat)
    print(colored("Device:", "green", attrs=["bold"]), cfg.device)
    print(colored("Learnable parameters:", "green", attrs=["bold"]), f"{total_params}M")
    print(colored("Architecture:", "green", attrs=["bold"]), agent)

    def evaluate(step: int, episode_idx: int) -> dict:
        """Evaluate agent in eval_env and log metrics"""
        eval_metrics = {env_name: {} for env_name in env_names}
        eval_start_time = time.time()
        with torch.no_grad():
            episodic_returns, episodic_successes = {name: [] for name in env_names}, {
                name: [] for name in env_names
            }
            for _ in range(cfg.num_eval_episodes):
                eval_data = eval_env.rollout(
                    max_steps=cfg.max_episode_steps // cfg.action_repeat,
                    policy=eval_policy_module,
                    break_when_any_done=False,
                )

                success = eval_data["next"].get("success", None)
                for i, env_name in enumerate(env_names):
                    episodic_returns[env_name].append(
                        eval_data["next"]["episode_reward"][i][-1].cpu().item()
                    )

                    if success is not None:
                        episodic_successes[env_name].append(success[i].any())

            for i, env_name in enumerate(env_names):
                eval_episodic_return = (
                    sum(episodic_returns[env_name]) / cfg.num_eval_episodes
                )
                eval_metrics[env_name]["episodic_return"] = eval_episodic_return
            eval_episodic_return_mean = np.mean(
                [eval_metrics[env_name]["episodic_return"] for env_name in env_names]
            )

            if success is not None:
                # TODO is episodic_successes being calculated correctly
                episodic_success = sum(episodic_successes) / cfg.num_eval_episodes
                eval_metrics.update({"episodic_success": episodic_success})

        ##### Eval metrics #####
        eval_metrics.update(
            {
                "episodic_return_mean": eval_episodic_return_mean,
                "elapsed_time": time.time() - start_time,
                "SPS": int(step / (time.time() - start_time)),
                "episode_time": (time.time() - eval_start_time) / cfg.num_eval_episodes,
                "env_step": step * cfg.action_repeat,
                "step": step,
                "episode": episode_idx,
            }
        )

        if cfg.verbose:
            logger.info(
                f"Episode {episode_idx} | Env Step {step*cfg.action_repeat} | "
                f"Eval return (mean over envs) {eval_episodic_return_mean:.2f}"
            )

        when_to_log = [0, cfg.num_episodes // 2, cfg.num_episodes - 1]
        if cfg.capture_eval_video and episode_idx in when_to_log:
            with torch.no_grad():
                for video_env in video_envs:
                    video_env.rollout(
                        max_steps=cfg.max_episode_steps // cfg.action_repeat,
                        policy=eval_policy_module,
                        break_when_any_done=False,
                    )
                    video_env.transform.dump()

        ##### Log rank of latent and active codebook percent #####
        batch = rb.sample(batch_size=agent.encoder.cfg.latent_dim)
        eval_metrics.update(agent.metrics(batch))

        ##### Log metrics to W&B or csv #####
        writer.log_scalar(name="eval/", value=eval_metrics)
        return eval_metrics

    step = 0  # NOTE: 1 step means 1 step per sub-envs
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
        rb.extend(data)

        if episode_idx == 0:
            print(colored("First episodes data:", "green", attrs=["bold"]), data)

            # Evaluate the initial agent
            _ = evaluate(step=step, episode_idx=episode_idx)

        ##### Log episode metrics #####
        num_new_transitions = sum(
            data["next"]["step_count"][i][-1].cpu().sum().item()
            for i in range(env_count)
        )
        step += num_new_transitions
        episode_rewards = [
            data["next"]["episode_reward"][i][-1].cpu().item() for i in range(env_count)
        ]

        episodic_return_mean = sum(episode_rewards) / env_count
        if cfg.verbose:
            logger.info(
                f"Episode {episode_idx} | Env Step {step*cfg.action_repeat} | "
                f"Train return (mean over envs) {episodic_return_mean:.2f} | "
                f"Train return per env {' '.join(map(str, episode_rewards))}"
            )
        rollout_metrics = {
            "episodic_return_mean": episodic_return_mean,
            "episodic_length": num_new_transitions,
            "env_step": step * cfg.action_repeat,
        }
        rollout_metrics.update({env_name: {} for env_name in env_names})
        for i in range(env_count):
            rollout_metrics[env_names[i]]["episodic_return"] = episode_rewards[i]

        success = data["next"].get("success", None)
        if success is not None:
            episode_success = success.any()
            rollout_metrics.update({"episodic_success": episode_success})

        writer.log_scalar(name="rollout/", value=rollout_metrics)

        ##### Train agent (after collecting some random episodes) #####
        if episode_idx > cfg.random_episodes - 1:
            train_metrics = agent.update(
                replay_buffer=rb, num_new_transitions=num_new_transitions
            )

            ##### Log training metrics #####
            writer.log_scalar(name="train/", value=train_metrics)

            if episode_idx % 25 == 0:
                for i in range(env_count):
                    single_task_metrics = agent.fake_update(
                        replay_buffer=rb,
                        num_new_transitions=num_new_transitions,
                        rb_idx=i,
                    )
                    writer.log_scalar(
                        name=f"train_{env_names[i]}/", value=single_task_metrics
                    )

            ##### Save checkpoint #####
            torch.save({"model": agent.state_dict()}, "./checkpoint")

            ###### Evaluate ######
            if episode_idx % cfg.eval_every_episodes == 0:
                evaluate(step=step, episode_idx=episode_idx)

        # Release some GPU memory (if possible)
        torch.cuda.empty_cache()

    env.close()
    eval_env.close()


if __name__ == "__main__":
    cluster_safe_train()  # pyright: ignore
