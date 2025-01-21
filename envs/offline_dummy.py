#!/usr/bin/env python3
from typing import Optional

from tensordict import TensorDict, TensorDictBase
import torch
from torchrl.envs import EnvBase
from torchrl.data import Composite, Bounded, UnboundedContinuous


# TODO: Is this class worth it? If used for just one rollout we could also just use DMC
class OfflineDummyEnv(EnvBase):
    """A dummy env suitable for being overwritten with offline data"""

    def __init__(self, obs_dim: int, act_dim: int, device: str = "cpu"):
        super().__init__(device=device)

        self.obs_dim = obs_dim
        self.act_dim = act_dim

        self._make_spec()

    def _make_spec(self):
        self.observation_spec = Composite(
            state=UnboundedContinuous(shape=(self.obs_dim,), dtype=torch.float32)
        )

        self.action_spec = Bounded(
            low=torch.full((self.act_dim,), -1.0, dtype=torch.float32),
            high=torch.full((self.act_dim,), 1.0, dtype=torch.float32),
            shape=(self.act_dim,),
            dtype=torch.float32,
        )

        self.reward_spec = UnboundedContinuous(shape=(1,))

    def _set_seed(self, seed: Optional[int]):
        rng = torch.manual_seed(seed)
        self.rng = rng

    def _reset(self, tensordict: TensorDictBase, **kwargs):
        return TensorDict({"observation": torch.zeros(self.obs_dim)}, batch_size=[]).to(
            self.device
        )

    def _step(self, tensordict: TensorDictBase):
        return TensorDict(
            {
                "reward": torch.zeros(1),
                "observation": torch.zeros(self.obs_dim),
                "done": torch.tensor(False, dtype=torch.bool, device=self.device),
            },
            batch_size=[],
        ).to(self.device)
