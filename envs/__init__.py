#!/usr/bin/env python3
from typing import Optional

import gymnasium as gym
from dm_control import suite
from tensordict import TensorDictBase
import torch
from torchrl.data.tensor_specs import Bounded
from torchrl.envs import GymEnv, StepCounter, TransformedEnv
from torchrl.envs.transforms import (
    CatFrames,
    Compose,
    DoubleToFloat,
    RenameTransform,
    Resize,
    RewardSum,
    ToTensorImage,
    Transform,
    TransformedEnv,
)
from torchrl.record import VideoRecorder
from torchrl.record.loggers import WandbLogger

from .dmcontrol import make_env as dmcontrol_make_env


class BodyAndTaskIDs(Transform):
    """A transform to add one-hot encoded body and/or task IDs to an env."""

    def __init__(
        self,
        body_id: Optional[torch.Tensor] = None,
        task_id: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        for item in [body_id, task_id]:
            if item is None:
                continue
            assert torch.all((item == 0) | (item == 1)), "One-hot values must be binary"
            assert item.sum() == 1, "One-hot values must sum to 1"
        self.body_id = body_id.float() if body_id is not None else None
        self.task_id = task_id.float() if task_id is not None else None

    def _call(self, tensordict: TensorDictBase) -> TensorDictBase:
        if self.body_id is not None:
            tensordict["observation"].set("body_id", self.body_id)
        if self.task_id is not None:
            tensordict["observation"].set("task_id", self.task_id)
        return tensordict

    def _reset(
        self, tensordict: TensorDictBase, tensordict_reset: TensorDictBase
    ) -> TensorDictBase:
        return self._call(tensordict_reset)

    def transform_observation_spec(self, observation_spec):
        if self.body_id is not None:
            observation_spec["observation"]["body_id"] = Bounded(
                low=0, high=1, shape=self.body_id.shape, dtype=torch.float
            )
        if self.task_id is not None:
            observation_spec["observation"]["task_id"] = Bounded(
                low=0, high=1, shape=self.task_id.shape, dtype=torch.float
            )
        return observation_spec

    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        raise RuntimeError("BodyAndTaskIDs can only be used with a transformed env")


class TaskMasker(Transform):
    """A transform to add a mask to an env, indicating relevant obs and action dims."""

    def __init__(
        self,
        obs_dim: Optional[int],
        act_dim: Optional[int],
        max_obs_dim: Optional[int],
        max_act_dim: Optional[int],
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.max_obs_dim = max_obs_dim
        self.max_act_dim = max_act_dim

        self.obs_mask = None
        self.act_mask = None
        if self.obs_dim is not None and self.max_obs_dim is not None:
            self.obs_mask = self._make_mask(self.obs_dim, self.max_obs_dim)
        if self.act_dim is not None and self.max_act_dim is not None:
            self.act_mask = self._make_mask(self.act_dim, self.max_act_dim)

    def _make_mask(self, relevant_dim: int, mask_dim: int):
        assert relevant_dim <= mask_dim
        mask = torch.zeros(mask_dim, dtype=torch.float32)
        mask[:relevant_dim] = 1.0
        return mask

    def _call(self, tensordict):
        if self.obs_mask is not None:
            tensordict["observation"].set("obs_mask", self.obs_mask)
        if self.act_mask is not None:
            tensordict["observation"].set("act_mask", self.act_mask)
        return tensordict

    def _reset(
        self, tensordict: TensorDictBase, tensordict_reset: TensorDictBase
    ) -> TensorDictBase:
        return self._call(tensordict_reset)

    def transform_observation_spec(self, observation_spec):
        if self.obs_mask is not None:
            observation_spec["observation"]["obs_mask"] = Bounded(
                low=0, high=1, shape=(self.max_obs_dim,), dtype=torch.float
            )
        if self.act_dim is not None:
            observation_spec["observation"]["act_mask"] = Bounded(
                low=0, high=1, shape=(self.max_act_dim,), dtype=torch.float
            )
        return observation_spec

    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        raise RuntimeError("TaskMasker can only be used with a transformed env")


def make_env(
    env_name: str,
    task_name: Optional[str] = None,
    body_id: Optional[torch.Tensor] = None,
    task_id: Optional[torch.Tensor] = None,
    obs_dim: Optional[int] = None,
    act_dim: Optional[int] = None,
    max_obs_dim: Optional[int] = None,
    max_act_dim: Optional[int] = None,
    use_offline_data: bool = False,
    seed: int = 42,
    from_pixels: bool = True,
    frame_skip: int = 2,
    pixels_only: bool = False,
    render_size: int = 64,
    num_frames_to_stack: int = 1,
    logger=None,
    record_video: bool = False,
    device: str = "cpu",
):
    if not from_pixels:
        pixels_only = False

    if use_offline_data:
        env = make_offline_env(env_name=env_name, task_name=task_name, device=device)
    else:
        if env_name in gym.envs.registry.keys():
            env = GymEnv(
                env_name=env_name,
                from_pixels=from_pixels,
                frame_skip=frame_skip,
                pixels_only=pixels_only,
                device=device,
            )
        elif (env_name, task_name) in suite.ALL_TASKS or env_name == "cup":
            env = make_dmcontrol_env(
                env_name=env_name,
                task_name=task_name,
                from_pixels=from_pixels or record_video,
                frame_skip=frame_skip,
                pixels_only=pixels_only,
                device=device,
            )

    transforms = []
    if not pixels_only:
        transforms.append(RenameTransform(in_keys=["observation"], out_keys=["state"]))
        transforms.append(
            RenameTransform(in_keys=["state"], out_keys=[("observation", "state")])
        )
    transforms.append(DoubleToFloat())
    transforms.append(StepCounter())
    transforms.append(RewardSum())
    transforms.append(BodyAndTaskIDs(body_id, task_id))
    if max_obs_dim is not None or max_act_dim is not None:
        tm = TaskMasker(
            obs_dim=obs_dim,
            act_dim=act_dim,
            max_obs_dim=max_obs_dim,
            max_act_dim=max_act_dim,
        )
        transforms.append(tm)

    if from_pixels:
        transforms.append(ToTensorImage(in_keys="pixels"))
        transforms.append(Resize(render_size, render_size))
        transforms.append(
            RenameTransform(in_keys=["pixels"], out_keys=[("observation", "pixels")])
        )
        transforms.append(
            CatFrames(N=num_frames_to_stack, dim=-3, in_keys=("observation", "pixels"))
        )
        video_rec_in_keys = ("observation", "pixels")
    else:
        video_rec_in_keys = "pixels"

    if record_video:
        if logger is None:
            logger = WandbLogger(exp_name="", log_dir="./logs")
        transforms.append(
            VideoRecorder(
                logger=logger,
                tag=f"run_video_{env_name}-{task_name}",
                in_keys=video_rec_in_keys,
            )
        )

    env = TransformedEnv(env, Compose(*transforms))
    env.set_seed(seed)

    return env
