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
            tensordict["observation"]["body_id"] = self.body_id
        if self.task_id is not None:
            tensordict["observation"]["task_id"] = self.task_id
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


def make_env(
    env_name: str,
    task_name: Optional[str] = None,
    body_id: Optional[torch.Tensor] = None,
    task_id: Optional[torch.Tensor] = None,
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

    if env_name in gym.envs.registry.keys():
        env = GymEnv(
            env_name=env_name,
            from_pixels=from_pixels,
            frame_skip=frame_skip,
            pixels_only=pixels_only,
            device=device,
        )
    elif (env_name, task_name) in suite.ALL_TASKS or env_name == "cup":
        env = dmcontrol_make_env(
            env_name=env_name,
            task_name=task_name,
            from_pixels=from_pixels or record_video,
            frame_skip=frame_skip,
            pixels_only=pixels_only,
            device=device,
        )

    if not pixels_only:
        env = TransformedEnv(
            env,
            Compose(
                RenameTransform(in_keys=["observation"], out_keys=["state"]),
                RenameTransform(in_keys=["state"], out_keys=[("observation", "state")]),
            ),
        )
    env = TransformedEnv(
        env,
        Compose(
            DoubleToFloat(),
            StepCounter(),
            RewardSum(),
            BodyAndTaskIDs(body_id, task_id),
        ),
    )

    if from_pixels:
        env = TransformedEnv(
            env,
            Compose(
                ToTensorImage(in_keys="pixels"),
                Resize(render_size, render_size),
                # RenameTransform(in_keys="pixels", out_keys=("observation", "pixels")),
                RenameTransform(
                    in_keys=["pixels"], out_keys=[("observation", "pixels")]
                ),
                CatFrames(
                    N=num_frames_to_stack, dim=-3, in_keys=("observation", "pixels")
                ),
            ),
        )
        video_rec_in_keys = ("observation", "pixels")
    else:
        video_rec_in_keys = "pixels"

    if record_video:
        if logger is None:
            logger = WandbLogger(exp_name="", log_dir="./logs")
        env = TransformedEnv(
            env,
            VideoRecorder(
                logger=logger,
                tag=f"run_video_{env_name}-{task_name}",
                in_keys=video_rec_in_keys,
            ),
        )
    env.set_seed(seed)
    return env
