#!/usr/bin/env python3
from typing import Optional

import gymnasium as gym
from dm_control import suite
from tensordict import TensorDictBase
import torch
from torchrl.data.tensor_specs import Bounded, Categorical, Composite
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

# NOTE: Transform._call says it's called by step() and reset() but only step() is true
# That's why _reset is needed below. See also: https://github.com/pytorch/rl/issues/2595


class BodyAndTaskIDs(Transform):
    """A transform to add one-hot encoded body and/or task IDs to an env."""

    def __init__(
        self,
        body_id: Optional[torch.Tensor] = None,
        task_id: Optional[torch.Tensor] = None,
        n_body: Optional[int] = None,
        n_task: Optional[int] = None,
    ):
        super().__init__()
        self.n_body = n_body
        self.n_task = n_task
        self.body_id = body_id
        self.task_id = task_id

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

    def transform_observation_spec(self, observation_spec: Composite) -> Composite:
        if self.body_id is not None:
            observation_spec["observation"]["body_id"] = Categorical(
                n=self.n_body,
                shape=self.body_id.shape,
                dtype=torch.float,
                device=observation_spec.device,
            )
        if self.task_id is not None:
            observation_spec["observation"]["task_id"] = Categorical(
                n=self.n_task,
                shape=self.task_id.shape,
                dtype=torch.float,
                device=observation_spec.device,
            )
        return observation_spec

    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        raise RuntimeError("BodyAndTaskIDs can only be used with a transformed env")


class TaskMasker(Transform):
    """A transform to add a mask to an env, indicating relevant action dims."""

    def __init__(self, act_dim: int, max_act_dim: int):
        super().__init__()

        assert act_dim <= max_act_dim
        self.max_act_dim = max_act_dim

        self.act_mask = torch.zeros(max_act_dim, dtype=torch.float32)
        self.act_mask[:act_dim] = 1.0

    def _call(self, tensordict):
        tensordict["observation"].set("act_mask", self.act_mask)
        return tensordict

    def _reset(
        self, tensordict: TensorDictBase, tensordict_reset: TensorDictBase
    ) -> TensorDictBase:
        return self._call(tensordict_reset)

    def transform_observation_spec(self, observation_spec: Composite) -> Composite:
        observation_spec["observation"]["act_mask"] = Bounded(
            low=0,
            high=1,
            shape=(self.max_act_dim,),
            dtype=torch.float,
            device=observation_spec.device,
        )
        return observation_spec

    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        raise RuntimeError("TaskMasker can only be used with a transformed env")


def make_env(
    env_name: str,
    task_name: Optional[str] = None,
    body_id: Optional[torch.Tensor] = None,
    task_id: Optional[torch.Tensor] = None,
    n_body: Optional[int] = None,
    n_task: Optional[int] = None,
    obs_dim: Optional[int] = None,
    act_dim: Optional[int] = None,
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

    elif env_name in gym.envs.registry.keys():
        env = GymEnv(
            env_name=env_name,
            from_pixels=from_pixels,
            frame_skip=frame_skip,
            pixels_only=pixels_only,
            device=device,
        )
    elif (env_name, task_name) in suite.ALL_TASKS or env_name in ["cup", "swimmer"]:
        env = dmcontrol_make_env(
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
    transforms.append(
        BodyAndTaskIDs(body_id=body_id, task_id=task_id, n_body=n_body, n_task=n_task)
    )
    if act_dim is not None and max_act_dim is not None:
        transforms.append(TaskMasker(act_dim=act_dim, max_act_dim=max_act_dim))

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
