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

    # Experiment
    use_offline_data: bool = False  # Train fully offline (but evaluate still online)
    normalize_states: bool = False  # Only takes effect if use_offline_data==True
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
    eval_only: bool = False  # Skip training (useful when loading checkpoint)
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
    from functools import partial

    from hydra.core.hydra_config import HydraConfig
    import numpy as np
    from termcolor import colored
    from tensordict import LazyStackedTensorDict, TensorDict, pad_sequence
    from tensordict.nn import TensorDictModule
    from torchrl.data.tensor_specs import BoundedContinuous
    from torchrl.envs import ParallelEnv
    from torchrl.record.loggers.wandb import WandbLogger
    import torch

    from envs import make_env
    from iqrl import iQRL
    import utils.helper as h
    from utils import ReplayBuffer

    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger(__name__)

    assert cfg.agent.obs_types == ["state"], "only obs_types=['state'] is supported"
    assert not cfg.eval_only or cfg.checkpoint is not None, "eval_only needs checkpoint"

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
    body_str_to_id = h.seq_to_1hot([body_name for body_name, _ in cfg.envs])
    task_str_to_id = h.seq_to_1hot([task_name for _, task_name in cfg.envs])

    common_kwargs_for_make_env = {
        "seed": cfg.seed,
        "frame_skip": cfg.action_repeat,
        "from_pixels": False,
        "pixels_only": False,
        "logger": writer,
    }
    create_fn = [
        partial(
            make_env,
            env_name=body_name,
            task_name=task_name,
            body_id=body_str_to_id[body_name],
            task_id=task_str_to_id[task_name],
            record_video=False,  # No need, video_envs below will record
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
        partial(
            fn, obs_dim=od[i], act_dim=ad[i], max_obs_dim=max(od), max_act_dim=max(ad)
        )
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
    video_envs = [
        make_env(
            env_name=body_name,
            task_name=task_name,
            body_id=body_str_to_id[body_name],
            task_id=task_str_to_id[task_name],
            record_video=cfg.capture_eval_video,
            use_offline_data=False,
            **common_kwargs_for_make_env,
        )
        for body_name, task_name in cfg.envs
    ]

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
    ids_to_dims = {}  # (body ID, task ID) -> (obs dim, action dim), all integers
    for i in range(env_count):
        body_id = np.argmax(body_str_to_id[cfg.envs[i][0]]).item()
        task_id = np.argmax(task_str_to_id[cfg.envs[i][1]]).item()
        o = np.array(obs_specs[i]["state"].shape).prod().item()
        a = np.array(act_specs[i].shape).prod().item()
        ids_to_dims[(body_id, task_id)] = (o, a)
    agent = iQRL(
        cfg=cfg.agent, obs_specs=obs_specs, act_specs=act_specs, ids_to_dims=ids_to_dims
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

    def evaluate(step: int, episode_idx: int, start_time: float) -> dict:
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

    if cfg.use_offline_data:
        print("Loading offline data into replay buffer...")

        rollout_blueprint = env.rollout(
            max_steps=cfg.max_episode_steps // cfg.action_repeat,
            policy=policy_module,
            break_when_any_done=True,  # Same-length episodes -> break when done
        )

        offline_data = []
        for i in range(env_count):
            MT30_DATA_DIR = os.path.join(
                os.environ.get("WRKDIR"), "data", "mt30", "per-task"
            )
            file_path = os.path.join(os.path.join(MT30_DATA_DIR, f"{env_names[i]}.pt"))
            task_data_raw = torch.load(file_path, weights_only=False).to(cfg.device)
            assert sorted(task_data_raw.keys()) == ["action", "obs", "reward"]

            MAX_EXP = 1000
            task_data_raw = task_data_raw[:MAX_EXP]  # Avoid too much memory usage

            task_blueprint = rollout_blueprint[i]
            new_shape = [task_data_raw.shape[0]] + list(task_blueprint.shape)

            task_data = task_blueprint.unsqueeze(0).expand(new_shape)
            task_data["observation"]["state"] = task_data_raw["obs"][:, :-1]
            task_data["next"]["observation"]["state"] = task_data_raw["obs"][:, 1:]
            task_data["action"] = task_data_raw["action"][:, 1:]
            task_data["reward"] = task_data_raw["reward"][:, 1:]
            # NOTE: "done" and ("next", "terminated") can remain False all the way, it's fine

            offline_data.append(task_data)

        data = LazyStackedTensorDict.lazy_stack(offline_data, dim=0)
        data = pad_sequence(data, pad_dim=-1)  # LazyStackedTensorDict -> TensorDict
        data = data.flatten(0, 1)  # Merge first 2 dims: env_count and episode_count

        if cfg.normalize_states:
            # Normalize over all states (even across different tasks)
            eps = 1e-3
            # States have shape: (episode_count, ep_length, max_state_dim)
            mean = data["observation"]["state"].mean(dim=(0, 1), keepdims=True)
            std = data["observation"]["state"].std(dim=(0, 1), keepdims=True) + eps
            data["observation"]["state"] = (data["observation"]["state"] - mean) / std
            data["next"]["observation"]["state"] = (
                data["next"]["observation"]["state"] - mean
            ) / std
            print(f"Normalized over all states:\n\tmean={mean}\n\tstd={std})")

        rb.extend(data)

    step = 0  # NOTE: 1 step means 1 step per sub-envs
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

        if episode_idx == 0:
            if not cfg.use_offline_data:
                print(colored("First episodes data:", "green", attrs=["bold"]), data)

            # Evaluate the initial agent
            _ = evaluate(
                cfg,
                step=step,
                episode_idx=episode_idx,
                start_time=start_time,
            )
            if cfg.eval_only:
                break  # Eval is done; close envs and exit

        if not cfg.use_offline_data:
            ##### Log episode metrics #####
            num_new_transitions = sum(
                data["next"]["step_count"][i][-1].cpu().sum().item()
                for i in range(env_count)
            )
            step += num_new_transitions
            episode_rewards = [
                data["next"]["episode_reward"][i][-1].cpu().item()
                for i in range(env_count)
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
                "episodic_length": num_new_transitions // env_count,
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
        else:
            num_new_transitions = 500
            step += num_new_transitions

        ##### Train agent (after collecting some random episodes) #####
        if cfg.use_offline_data or episode_idx > cfg.random_episodes - 1:
            train_metrics = agent.update(
                replay_buffer=rb, num_new_transitions=num_new_transitions
            )
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

            torch.save({"model": agent.state_dict()}, "./checkpoint")

            if episode_idx % cfg.eval_every_episodes == 0:
                evaluate(step=step, episode_idx=episode_idx, start_time=start_time)

        # Release some GPU memory (if possible)
        torch.cuda.empty_cache()

    env.close()
    eval_env.close()


if __name__ == "__main__":
    cluster_safe_train()  # pyright: ignore
