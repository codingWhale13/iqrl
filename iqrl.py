#!/usr/bin/env python3
import copy
import logging
import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils
import utils.helper as h
import wandb
from tensordict import LazyStackedTensorDict, TensorDict, TensorDictBase
from torchrl.data import Bounded, CompositeSpec
from utils import ReplayBuffer, ReplayBufferSamples


logger = logging.getLogger(__name__)


@dataclass
class iQRLConfig:
    """Config for iQRL"""

    """Strategy for state and action spaces of differently sized dimensionality"""
    state_action_mode: str = "padding"
    """Map environment states to latent states before using them in other components"""
    use_obs_encoder: bool = True  # Used in original iQRL, thus defaults to True
    """Map policy actions to latent actions before using them in dynamics and critic"""
    use_action_encoder: bool = False  # Not in original iQRL, thus defaults to False
    """Condition {encoders, transition dynamics, actor, critic} on body & task IDs"""
    condition_encoders: bool = True
    condition_dynamics: bool = True
    condition_actor: bool = True
    condition_critic: bool = True
    """MLP dims for actor/critic/dynamics"""
    mlp_dims: List[int] = field(default_factory=lambda: [1024, 1024])
    """Learning rate for actor/critic"""
    lr: float = 3e-4
    """Batch size - same for for representation learning and actor/critic"""
    batch_size: int = 256
    """Number of parameter updates per new data, i.e. UTD ratio """
    utd_ratio: int = 1
    """Update actor less frequently than critic"""
    actor_update_freq: int = 2
    """Discount factor"""
    gamma: float = 0.99
    """Target network update rate"""
    tau: float = 0.005
    """Number of critics"""
    num_critics: int = 5
    """Number of critics to sample"""
    q_sample_size: int = 2
    """Use N-step returns for Q-learning?"""
    nstep: int = 1  # nstep returns
    """What observation types to use? ["state"] or ["pixels"] or ["state", "pixels"]"""
    obs_types: List[str] = field(default_factory=lambda: ["state"])

    """ENCODER CONFIG"""
    """Size of latent state space"""
    latent_dim: int = 512
    """Size of latent action space is the largest action dim multiplied by this"""
    latent_action_dim_factor: int = 1
    """Horizon used for representation learning"""
    horizon: int = 5
    """Discount factor for representation learning"""
    rho: float = 0.9
    """MLP dims for encoder/decoder"""
    enc_mlp_dims: List[int] = field(default_factory=lambda: [256])
    """Learning rate for encoder/dynamics/projection/reward"""
    enc_lr: float = 1e-4
    """Momentum coefficient for target encoder"""
    enc_tau: float = 0.005
    """Update encoder less frequently than actor/critic"""
    enc_update_freq: int = 1
    """Clips the gradient norm of the encoder"""
    grad_clip_norm: Optional[float] = 20
    """Use target encoder for representation learning"""
    use_tar_enc: bool = True
    """Predict change in latent or next latent? i.e. next_z = z + f(z, a) else next_z = f(z, a)"""
    use_delta: bool = True
    """Use LayerNorm or BatchNorm for encoder?"""
    enc_norm_type: str = "ln"
    """(Optionally) use dropout for critic"""
    q_dropout: float = 0.0
    """(Optionally) use dropout for MLP encoder"""
    enc_dropout: float = 0.0
    """Use temporal consistency loss for representation learning"""
    use_tc_loss: bool = True
    """Use reward prediction for representation learning"""
    use_rew_loss: bool = False
    """Use cosine similarity for consistency loss - otherwise MSE"""
    use_cosine_similarity_dynamics: bool = True
    """Flag to turn FSQ on/off """
    use_fsq: bool = True
    """FSQ levels - setting as [8,8] corresponds to a codebook of size 8*8=62=2^8"""
    fsq_levels: List[int] = field(default_factory=lambda: [8, 8])
    """Use offline data to train and use TD3-BC instead of TD3"""
    use_offline_data: bool = "${use_offline_data}"  # Set from TrainConfig
    """States are normalized per-task"""
    normalize_states: bool = "${normalize_states}"  # Set from TrainConfig
    """When using offline data, this is the only additional parameter (see TD3-BC)"""
    bc_alpha: float = 2.5

    """PROJECTION HEAD"""
    """Project the latent using an MLP before calculating the temporal consistency loss?"""
    use_latent_projection: bool = False
    """MLP dims for projection head"""
    projection_mlp_dims: List[int] = field(default_factory=lambda: [256])
    """Dimension of projection - defaults to latent_dim/16"""
    proj_dim: Optional[int] = None

    """EXPLORATION NOISE SCHEDULE"""
    """Initial variance"""
    exploration_noise_start: float = 1.0
    """Final variance"""
    exploration_noise_end: float = 0.1
    """Number of episodes do decay noise"""
    exploration_noise_num_steps: int = 50

    """POLICY SMOOTHING"""
    """Variance"""
    policy_noise: float = 0.2
    """Clip the noise"""
    noise_clip: float = 0.3

    """OTHER"""
    """Call wandb.log() while updating the agent (creates "Charts" section in wandb)"""
    log_during_update: bool = False  # Avoid memory-intensive logging by default
    """Logging frequency, only takes effect if log_during_update==True"""
    logging_freq: int = 100
    """If True try to compile all NNs"""
    compile: bool = False
    """All NNs will be put on this device"""
    device: str = "${device}"  # Set from TrainConfig
    """Print training losses?"""
    verbose: bool = "${verbose}"  # Set from TrainConfig


class Actor(nn.Module):
    def __init__(
        self,
        cfg: iQRLConfig,
        in_dim: int,
        act_dim: int,
        action_scale: float,
        action_bias: float,
    ):
        super().__init__()
        self.cfg = cfg
        self.action_scale = action_scale
        self.action_bias = action_bias
        self.mlp = h.mlp(in_dim, self.cfg.mlp_dims, act_dim)

    def forward(self, s: torch.Tensor, ids: list[torch.Tensor]):
        x = torch.cat([s] + ids, -1) if self.cfg.condition_actor else s
        a = self.mlp(x)
        a = torch.tanh(a)
        a = a * self.action_scale + self.action_bias
        return a


class Critic(nn.Module):
    def __init__(self, cfg: iQRLConfig, in_dim: int):
        super().__init__()
        self.cfg = cfg

        qs = [
            h.mlp(
                in_dim=in_dim,
                mlp_dims=cfg.mlp_dims,
                out_dim=1,
                dropout=cfg.q_dropout,
            ).to(cfg.device)
            for _ in range(cfg.num_critics)
        ]
        for q in qs:
            h.orthogonal_init(q.parameters())

        self.qs = h.Ensemble(qs)

    def forward(
        self,
        s: torch.Tensor,
        a: torch.Tensor,
        ids: list[torch.Tensor],
        return_type: str = "all",
    ):
        x = torch.cat([s, a] + ids if self.cfg.condition_critic else [s, a], -1)
        qs = self.qs(x)
        if return_type == "all":
            return qs

        # Sample two Q values
        if self.cfg.q_sample_size is not None:
            idxs = torch.randperm(qs.shape[0])[: self.cfg.q_sample_size]
            qs = qs[idxs]

        if return_type == "min":
            return torch.min(qs, 0)[0]
        elif return_type == "avg":
            return torch.mean(qs, 0)
        else:
            raise NotImplementedError(
                f"return_type should be 'all' or 'min' or 'avg' not {return_type}"
            )


class Encoder(nn.Module):
    def __init__(
        self,
        cfg: iQRLConfig,
        obs_dim: int,
        act_dim: int,
        latent_obs_dim: int,
        latent_act_dim: int,
        ids_dim: int,
    ):
        super().__init__()
        self.cfg = cfg
        self.obs_dim = obs_dim

        ##### Configure FSQ stuff #####
        if cfg.use_fsq:
            self._fsq = h.FSQ(levels=cfg.fsq_levels)

        ##### Init encoders #####
        self._encoder = nn.ModuleDict()
        if "state" in cfg.obs_types and cfg.use_obs_encoder:
            self._encoder["state"] = h.mlp(
                in_dim=obs_dim + (ids_dim if cfg.condition_encoders else 0),
                mlp_dims=cfg.enc_mlp_dims,
                out_dim=latent_obs_dim,
                dropout=cfg.enc_dropout,
            )
        if cfg.use_action_encoder:
            self._encoder["action"] = h.mlp(
                in_dim=act_dim + (ids_dim if cfg.condition_encoders else 0),
                mlp_dims=cfg.enc_mlp_dims,
                out_dim=latent_act_dim,
                dropout=cfg.enc_dropout,
            )
        if cfg.use_tar_enc:
            self._encoder_tar = copy.deepcopy(self._encoder).requires_grad_(False)

        ##### Init dynamics transition model #####
        self._trans = h.mlp(
            in_dim=latent_obs_dim
            + latent_act_dim
            + (ids_dim if cfg.condition_dynamics else 0),
            mlp_dims=cfg.mlp_dims,
            out_dim=latent_obs_dim,
        )

        ##### Init optional models #####
        if cfg.use_latent_projection:
            if cfg.proj_dim is None:
                cfg.proj_dim = int(latent_obs_dim / 16)
            self._proj = h.mlp(latent_obs_dim, cfg.mlp_dims, cfg.proj_dim)
            if cfg.use_tar_enc:
                self._proj_tar = copy.deepcopy(self._proj).requires_grad_(False)

        if cfg.use_rew_loss:
            self._reward = h.mlp(latent_obs_dim + latent_act_dim, cfg.mlp_dims, 1)

    def encode_obs(self, obs: TensorDictBase, tar: bool = False) -> TensorDictBase:
        if not self.cfg.use_obs_encoder:
            return obs  # Identity mapping

        if "pixels" in self.cfg.obs_types:
            raise NotImplementedError()
        zs = {}
        if self.cfg.state_action_mode == "padding":
            if isinstance(obs, LazyStackedTensorDict):
                obs_tensor = obs.get_nestedtensor("state").to_padded_tensor(padding=0.0)
            else:
                obs_tensor = obs["state"]
            p1d = (0, self.obs_dim - obs_tensor.shape[-1])  # No assumptions for obs
            obs_padded = F.pad(obs_tensor, p1d, "constant", 0.0).to(self.cfg.device)
            if self.cfg.condition_encoders:
                ids = h.get_ids(obs=obs, device=self.cfg.device)
                obs_padded = torch.cat(ids + [obs_padded], dim=-1)
            if tar:
                zs["state"] = self._encoder_tar["state"](obs_padded)
            else:
                zs["state"] = self._encoder["state"](obs_padded)
        elif self.cfg.state_action_mode == "multi-head":
            raise NotImplementedError()
        elif self.cfg.state_action_mode == "attention":
            raise NotImplementedError()

        if "state" in self.cfg.obs_types and "pixels" not in self.cfg.obs_types:
            z = zs["state"]
            td = TensorDict({"state": z}, batch_size=obs.batch_size)
        elif "state" not in self.cfg.obs_types and "pixels" in self.cfg.obs_types:
            z = zs["pixels"]
        else:
            raise NotImplementedError("Need to make encoder take both state and pixels")

        td = TensorDict({"state": z}, batch_size=obs.batch_size)
        if self.cfg.use_fsq:
            td.update(self.quantize(z))
        return td

    def encode_action(
        self, action: torch.Tensor, ids: list[torch.Tensor], tar: bool = False
    ) -> torch.Tensor:
        if not self.cfg.use_action_encoder:
            return action  # Identity mapping

        # NOTE: No padding required because action comes from padded replay buffer
        if self.cfg.condition_encoders:
            action = torch.cat(ids + [action], dim=-1)

        if tar:
            za = self._encoder_tar["action"](action)
        else:
            za = self._encoder["action"](action)

        return za  # NOTE: actions are not being quantized currently

    def trans(self, s: torch.Tensor, a: torch.Tensor, ids: list[torch.Tensor]):
        sa = torch.concat(([s, a] + ids if self.cfg.condition_dynamics else [s, a]), -1)
        delta_s = self._trans(sa)
        next_s = s + delta_s if self.cfg.use_delta else delta_s
        return next_s

    def reward(self, s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        sa = torch.concat([s, a], -1)
        r = self._reward(sa)
        return r

    def project(self, s: torch.Tensor, tar: bool = False) -> torch.Tensor:
        """Project (maybe latent) state before calculating consistency loss"""
        s = self._proj_tar(s) if tar else self._proj(s)
        return s

    def quantize(self, z: torch.Tensor) -> dict[str, torch.Tensor]:
        """Quantize the latent state"""
        return self._fsq(z)

    def loss(self, batch: ReplayBufferSamples) -> Tuple[torch.Tensor, dict]:
        tc_loss = torch.zeros(1).to(self.cfg.device)
        reward_loss = torch.zeros(1).to(self.cfg.device)

        ##### Create targets #####
        ids = h.get_ids(obs=batch.observations, device=self.cfg.device)
        with torch.no_grad():
            states_tar = self.encode_obs(batch.next_observations, tar=True)["state"]
            actions = self.encode_action(batch.actions, ids=ids, tar=False)  # NO target

        ##### Latent rollout #####
        ids_t = h.get_ids(obs=batch.observations[0], device=self.cfg.device)
        states_rollout = torch.empty_like(states_tar)
        s = self.encode_obs(batch.observations[0])["state"]
        dones = torch.zeros_like(batch.dones[0], dtype=torch.bool)
        terminateds_or_dones = torch.zeros_like(batch.dones, dtype=torch.bool)
        for t in range(self.cfg.horizon):
            dones = torch.where(terminateds_or_dones[t], dones, batch.dones[t])
            terminateds_or_dones[t] = torch.logical_or(
                terminateds_or_dones[t], torch.logical_or(dones, batch.terminateds[t])
            )
            # Predict next (maybe latent) state
            next_s_pred = self.trans(s=s, a=actions[t], ids=ids_t)
            if self.cfg.use_fsq:
                next_s_pred = self.quantize(next_s_pred)["state"]
            s = next_s_pred
            states_rollout[t] = s

        rho = torch.tensor([self.cfg.rho**t for t in range(self.cfg.horizon)]).to(
            self.cfg.device
        )
        terminateds_or_dones = terminateds_or_dones.to(torch.int)

        ##### (Optional) Reward prediction loss #####
        if self.cfg.use_rew_loss:
            r_tar = batch.rewards[..., None]  # Reward target
            r_pred = self.reward(s=states_rollout, a=actions)
            assert r_pred.ndim == 3 and r_tar.ndim == 3
            _reward_loss = (r_pred[..., 0] - r_tar[..., 0]) ** 2
            _rho_reward_loss = rho * torch.mean(
                (1 - terminateds_or_dones) * _reward_loss, -1
            )
            reward_loss = torch.mean(_rho_reward_loss)

        ##### (Optional) Project latent before consistency loss #####
        if self.cfg.use_latent_projection:
            states_tar = self.project(states_tar, tar=True)
            states_rollout = self.project(states_rollout, tar=False)

        ##### Temporal consistency loss #####
        if self.cfg.use_tc_loss:
            if self.cfg.use_cosine_similarity_dynamics:
                """Cosine similarity"""
                _tc_loss = nn.CosineSimilarity(dim=-1, eps=1e-6)(
                    states_rollout, states_tar
                )
            else:
                """Mean squared error"""
                _tc_loss = torch.mean((states_rollout - states_tar) ** 2, dim=-1)
            _rho_tc_loss = rho * torch.mean((1 - terminateds_or_dones) * _tc_loss, -1)
            tc_loss = torch.mean(_rho_tc_loss)

        loss = tc_loss + reward_loss
        info = {
            "tc_loss": tc_loss.item(),
            "reward_loss": reward_loss.item(),
            "enc_loss": loss.item(),
            "z_min": torch.min(states_rollout).item(),
            "z_max": torch.max(states_rollout).item(),
            "z_mean": torch.mean(states_rollout.to(torch.float)).item(),
            "z_median": torch.median(states_rollout).item(),
        }
        return loss, info

    def metrics(self, batch: ReplayBufferSamples) -> dict:
        z = self.encode_obs(batch.observations[0])

        # Calculate rank of latent
        metrics = h.calc_rank(name="z", z=z["state"])

        # Calculate percentage of codebook that's active
        if self.cfg.use_fsq:
            num_codes = torch.tensor(math.prod(self.cfg.fsq_levels), device=z.device)

            def act_percent_fn(z):
                # TODO can't vmap this because Tensor.unique() not allowed in vmap
                return z.unique().numel() / num_codes * 100

            active_percents = torch.empty(z["indices"].shape[1])
            for i in range(z["indices"].shape[1]):
                active_percents[i] = act_percent_fn(z["indices"][i])
            metrics.update(
                {
                    # "active_percent": active_percent,
                    "active_percent_avg": active_percents.mean(),
                    "active_percent_min": active_percents.min(),
                    "active_percent_max": active_percents.max(),
                }
            )

        # TODO add dormant neuron ratio stuff
        # metrics.update(h.calc_dormant_neuron_ratio(batch, agent=self))

        return metrics

    def train(self):
        self._encoder.train()
        self._trans.train()
        if self.cfg.use_rew_loss:
            self._reward.train()
        if self.cfg.use_latent_projection:
            self._proj.train()

    def eval(self):
        self._encoder.eval()
        self._trans.eval()
        if self.cfg.use_rew_loss:
            self._reward.eval()
        if self.cfg.use_latent_projection:
            self._proj.eval()


class iQRL(nn.Module):
    def __init__(
        self,
        cfg: iQRLConfig,
        obs_specs: list[CompositeSpec],
        act_specs: list[Bounded],
        ids_to_dims: dict[tuple[int, int], tuple[int, int]],
    ):
        super().__init__()
        self.cfg = cfg
        self.ids_to_dims = ids_to_dims

        ##### Assert observation types (1d low and high; broadcasted later) #####
        assert len(obs_specs) == len(act_specs)
        act_spec_lo = act_specs[0].low[0]
        act_spec_hi = act_specs[0].high[0]
        # Make sure that the value ranges for actions are the same across all tasks
        for act_spec in act_specs:
            assert (act_spec.low == act_spec_lo).all(), "Inconsistent action range"
            assert (act_spec.high == act_spec_hi).all(), "Inconsistent action range"
        self.register_buffer("act_spec_low", act_spec_lo.to(cfg.device))
        self.register_buffer("act_spec_high", act_spec_hi.to(cfg.device))

        ##### Assert observation types #####
        if "pixels" in cfg.obs_types or "state" not in cfg.obs_types:
            raise NotImplementedError("Need to use state observations")

        ##### Prepare for FSQ (potentially changes latent dim -> do at beginning) #####
        if cfg.use_fsq:
            assert cfg.use_obs_encoder, "Can only use FSQ when using an obs encoder"
            num_channels = len(cfg.fsq_levels)
            if not cfg.latent_dim % num_channels == 0:
                raise NotImplementedError(
                    "latent_dim must be divisible by number of FSQ channels"
                )
            if num_channels > 1:
                logger.info(
                    f"Increasing latent dim from {cfg.latent_dim} to "
                    f"{cfg.latent_dim*num_channels} to account for FSQ channels"
                )
                cfg.latent_dim *= num_channels

        ##### Calculate max dims of observations, actions and body&task IDs #####
        obs_dim = max(np.prod(obs["state"].shape).item() for obs in obs_specs)
        self.act_dims = [np.prod(act_spec.shape).item() for act_spec in act_specs]
        act_dim = max(self.act_dims)
        ids_dim = sum(
            np.array(obs_specs[0][id_name].shape).prod().item()
            for id_name in ["body_id", "task_id"]
            if id_name in obs_specs[0].keys()
        )

        ##### Calculate dimensions of (optional) latent spaces #####
        latent_obs_dim = cfg.latent_dim if cfg.use_obs_encoder else obs_dim
        latent_act_dim = act_dim
        if cfg.use_action_encoder:
            latent_act_dim *= cfg.latent_action_dim_factor

        ##### Init encoders, dynamics, and optionally reward and projection models #####
        self.encoder = Encoder(
            cfg,
            obs_dim=obs_dim,
            act_dim=act_dim,
            latent_obs_dim=(latent_obs_dim),
            latent_act_dim=latent_act_dim,
            ids_dim=ids_dim,
        ).to(cfg.device)
        if cfg.compile:
            self.encoder = torch.compile(self.encoder, mode="default")
        self.enc_opt = torch.optim.AdamW(self.encoder.parameters(), lr=cfg.enc_lr)

        ##### Init actor network and its target network #####
        self._pi = Actor(
            cfg,
            in_dim=latent_obs_dim + (ids_dim if cfg.condition_actor else 0),
            act_dim=act_dim,
            action_scale=(act_spec_hi - act_spec_lo).to(cfg.device) / 2.0,
            action_bias=(act_spec_hi + act_spec_lo).to(cfg.device) / 2.0,
        ).to(cfg.device)
        self._pi = torch.compile(self._pi, mode="default") if cfg.compile else self._pi
        pi_tar = copy.deepcopy(self._pi).requires_grad_(False)
        self._pi_tar = torch.compile(pi_tar, mode="default") if cfg.compile else pi_tar

        ##### Init critics and their target networks #####
        Q = Critic(
            cfg,
            in_dim=latent_obs_dim
            + latent_act_dim
            + (ids_dim if cfg.condition_critic else 0),
        ).to(cfg.device)
        self.Q = torch.compile(Q, mode="default") if cfg.compile else Q
        Q_tar = copy.deepcopy(self.Q).requires_grad_(False)
        self.Q_tar = torch.compile(Q_tar, mode="default") if cfg.compile else Q_tar

        ##### Optimizers #####
        self.pi_opt = torch.optim.Adam(self._pi.parameters(), lr=cfg.lr)
        self.q_opt = torch.optim.Adam(self.Q.parameters(), lr=cfg.lr)

        ##### Exploration noise schedule #####
        self._exploration_noise_schedule = h.LinearSchedule(
            start=cfg.exploration_noise_start,
            end=cfg.exploration_noise_end,
            num_steps=cfg.exploration_noise_num_steps,
        )

        # Counters for number of param updates
        self.critic_update_counter = 0
        self.pi_update_counter = 0

        self.state_norm = None  # If cfg.normalize_states==True, will be set later

    def update(self, replay_buffer: ReplayBuffer, num_new_transitions: int) -> dict:
        """Update representation and TD3 at same time"""
        num_updates = int(num_new_transitions * self.cfg.utd_ratio)
        info = {}

        if self.cfg.verbose:
            logger.info(f"Performing {num_updates} iQRL updates...")
        for i in range(num_updates):
            batch = replay_buffer.sample()

            # Update enc less frequently than actor/critic
            if i % self.cfg.enc_update_freq == 0:
                info.update(self.representation_update_step(batch=batch))

            # Map observations and actions to latent
            with torch.no_grad():
                latent_obs = self.encoder.encode_obs(batch.observations, tar=False)
            batch = batch._replace(latent_obs=latent_obs)

            ##### Make nstep returns #####
            if self.cfg.horizon == 1:
                raise NotImplementedError("Check N-step batch is made correctly if h=1")
            nstep_batch = utils.to_nstep(
                batch, nstep=self.cfg.nstep, gamma=self.cfg.gamma
            )

            ##### Update critic #####
            info.update(self.critic_update_step(batch=nstep_batch))

            ##### Update actor less frequently than critic #####
            if self.critic_update_counter % self.cfg.actor_update_freq == 0:
                info.update(self.pi_update_step(batch=nstep_batch))

            if i % self.cfg.logging_freq == 0:
                if self.cfg.verbose:
                    logger.info(
                        f"Iteration {i} | loss {info['enc_loss']:.3} | tc loss {info['tc_loss']:.3} | reward loss {info['reward_loss']:.3}"
                    )
                if wandb.run is not None and self.cfg.log_during_update:
                    wandb.log(info)

        ###### Log some stuff ######
        info["exploration_noise"] = self.exploration_noise
        if wandb.run is not None and self.cfg.log_during_update:
            wandb.log({"exploration_noise": self.exploration_noise})

        self._exploration_noise_schedule.step()

        if self.cfg.verbose:
            logger.info("Finished training iQRL")
        return info

    def fake_update(
        self, replay_buffer: ReplayBuffer, num_new_transitions: int, rb_idx: int
    ) -> dict:
        """Fake update for logging, just to get info for single task"""
        num_updates = int(num_new_transitions * self.cfg.utd_ratio)
        info = {}

        for i in range(num_updates):
            batch = replay_buffer.sample()

            # Update enc less frequently than actor/critic
            if i % self.cfg.enc_update_freq == 0:
                info.update(self.representation_update_step(batch=batch, fake=True))

            # Map observations and actions to latent
            with torch.no_grad():
                latent_obs = self.encoder.encode_obs(batch.observations, tar=False)
            batch = batch._replace(latent_obs=latent_obs)

            ##### Make nstep returns #####
            if self.cfg.horizon == 1:
                raise NotImplementedError("Check N-step batch is made correctly if h=1")
            nstep_batch = utils.to_nstep(
                batch, nstep=self.cfg.nstep, gamma=self.cfg.gamma
            )

            ##### Update critic #####
            info.update(self.critic_update_step(batch=nstep_batch, fake=True))

            ##### Update actor less frequently than critic #####
            if self.critic_update_counter % self.cfg.actor_update_freq == 0:
                info.update(self.pi_update_step(batch=nstep_batch, fake=True))

        return info

    def representation_update_step(
        self, batch: ReplayBufferSamples, fake: bool = False
    ) -> dict:
        self.encoder.train()
        loss, info = self.encoder.loss(batch=batch)

        self.enc_opt.zero_grad(set_to_none=True)
        loss.backward()

        if self.cfg.grad_clip_norm is not None:
            enc_params = list(self.encoder.parameters())
            grad_norm = torch.nn.utils.clip_grad_norm_(
                enc_params, self.cfg.grad_clip_norm, error_if_nonfinite=False
            )
            info.update({"grad_norm": float(grad_norm)})

        if not fake:  # Actually perform the optimization step
            self.enc_opt.step()

            # Update the tar network
            h.soft_update_params(
                self.encoder._encoder, self.encoder._encoder_tar, tau=self.cfg.enc_tau
            )
            if self.cfg.use_latent_projection:
                h.soft_update_params(
                    self.encoder._proj, self.encoder._proj_tar, tau=self.cfg.enc_tau
                )

        self.encoder.eval()
        return info

    def critic_update_step(
        self, batch: ReplayBufferSamples, fake: bool = False
    ) -> dict:
        self.Q.train()
        self.Q_tar.train()

        # Check batch shapes
        assert batch.rewards.ndim == 1
        assert batch.rewards.shape[0] == batch.observations.shape[0]
        assert batch.latent_obs is not None

        # Make Q target
        ids = h.get_ids(obs=batch.observations, device=self.cfg.device)
        with torch.no_grad():
            s = batch.latent_obs["state"]
            next_s_raw = batch.next_observations
            next_s = self.encoder.encode_obs(next_s_raw, tar=False)["state"]

            a = self.encoder.encode_action(batch.actions, ids=ids)
            a_next_raw = (
                self.pi(next_s, ids=ids, tar=True, eval_mode=True, smooth=True)
                * batch.observations["act_mask"]
            )
            a_next = self.encoder.encode_action(a_next_raw, ids=ids)

            min_q_next_tar = self.Q_tar(s=next_s, a=a_next, ids=ids, return_type="min")
            min_q_next_tar = min_q_next_tar[..., 0]

            assert min_q_next_tar.shape == batch.rewards.shape
            next_q_value = (
                batch.rewards
                + (1 - batch.terminateds) * batch.next_state_gammas * min_q_next_tar
            )

        q_values = self.Q(s=s, a=a, ids=ids, return_type="all")[..., 0]
        next_q_value = next_q_value.broadcast_to(q_values.shape)
        q_loss = F.mse_loss(q_values, next_q_value)

        if not fake:  # Actually perform the optimization step
            self.critic_update_counter += 1

            ##### Optimize critic #####
            self.q_opt.zero_grad(set_to_none=True)
            q_loss.backward()
            self.q_opt.step()

            ##### Update the target network #####
            h.soft_update_params(self.Q, self.Q_tar, tau=self.cfg.tau)

        self.Q.eval()
        self.Q_tar.eval()
        info = {
            "q_loss": q_loss.item(),
            "q_mean": q_values.mean().item(),
            "q_min": q_values.min().item(),
            "q_max": q_values.max().item(),
            "q_std": q_values.std().item(),
            "q_targ_mean": next_q_value.mean().item(),
            "q_targ_min": next_q_value.min().item(),
            "q_targ_max": next_q_value.max().item(),
            "q_targ_std": next_q_value.std().item(),
            "critic_update_counter": self.critic_update_counter,
        }
        for i in range(self.cfg.num_critics):
            info.update({f"q{i+1}_values": q_values[i].mean().item()})
        return info

    def pi_update_step(self, batch: ReplayBufferSamples, fake: bool = False) -> dict:
        self.pi_update_counter += 1
        self._pi.train()

        assert batch.latent_obs is not None
        s = batch.latent_obs["state"]

        ids = h.get_ids(obs=batch.observations, device=self.cfg.device)
        pi_actions = self._pi(s=s, ids=ids) * batch.observations["act_mask"]
        pi_actions = self.encoder.encode_action(pi_actions, ids=ids)
        Q_values = self.Q(s=s, a=pi_actions, ids=ids, return_type="avg")

        if self.cfg.use_offline_data:
            # Add behavior cloning regularization
            lmbda = self.cfg.bc_alpha / Q_values.abs().mean().detach()
            pi_loss = -lmbda * Q_values.mean() + F.mse_loss(pi_actions, batch.actions)
        else:
            pi_loss = -Q_values.mean()

        if not fake:  # Actually perform the optimization step
            ##### Optimize actor #####
            self.pi_opt.zero_grad(set_to_none=True)
            pi_loss.backward()
            self.pi_opt.step()

            ##### Update the target network #####
            h.soft_update_params(self._pi, self._pi_tar, tau=self.cfg.tau)

        self._pi.eval()
        return {
            "actor_loss": pi_loss.item(),
            "actor_update_counter": self.pi_update_counter,
        }

    def set_state_norm(self, state_norm: dict):
        self.state_norm = state_norm  # (body_id, task_id)->(mean, std)

    @torch.no_grad()
    def select_action(
        self, obs: TensorDictBase, eval_mode: bool = False
    ) -> torch.Tensor:
        if self.cfg.normalize_states and self.state_norm is not None:
            # Normalize states (no need to pad though; this will be done by the encoder)
            use_nested_tensor = isinstance(obs, LazyStackedTensorDict)
            if use_nested_tensor:
                state = obs.get_nestedtensor("state")
            else:
                state = obs["state"]
            normalized_states = []
            num_states = state.size(0)  # This works for both tensor and nested tensor
            for i in range(num_states):
                body_id = np.argmax(obs["body_id"][i]).item()
                task_id = np.argmax(obs["task_id"][i]).item()
                mean, std = self.state_norm[(body_id, task_id)]
                normalized_states.append((state[i] - mean) / std)
            if use_nested_tensor:
                obs["state"] = torch.nested.nested_tensor(normalized_states)
            else:
                obs["state"] = torch.stack(normalized_states)

        is_flat_obs = False
        if obs.batch_size == torch.Size([]):
            obs = obs.view(1)
            is_flat_obs = True

        s = self.encoder.encode_obs(obs, tar=False).to(torch.float)
        ids = h.get_ids(obs=obs, device=self.cfg.device)
        a = self.pi(s["state"], ids, tar=False, eval_mode=eval_mode)

        if is_flat_obs:
            body_id = np.argmax(obs["body_id"][0]).item()
            task_id = np.argmax(obs["task_id"][0]).item()
            act_dim = self.ids_to_dims[(body_id, task_id)][1]
            return a[0][:act_dim]
        else:
            # NOTE: Assumes batched usage happens only during rollout, so order is known
            assert a.shape[0] == len(self.act_dims), "Batch size != number of subenvs"
            action_tensors = [a[i][:act_dim] for i, act_dim in enumerate(self.act_dims)]
            if len(set(self.act_dims)) == 1:
                td = torch.stack(action_tensors)  # Nested tensor gives error in rollout
            else:
                td = torch.nested.nested_tensor(action_tensors)  # Nested tensor works
            return td

    def pi(
        self,
        s: torch.Tensor,
        ids: list[torch.Tensor],
        tar: bool = False,
        eval_mode: bool = False,
        smooth: bool = False,
    ) -> torch.Tensor:
        a = self._pi_tar(s, ids) if tar else self._pi(s, ids)
        if not eval_mode:
            a += torch.normal(0, self._pi.action_scale * self.exploration_noise)
        if smooth:
            clipped_noise = (
                torch.randn_like(a, device=self.cfg.device) * self.cfg.policy_noise
            ).clamp(-self.cfg.noise_clip, self.cfg.noise_clip) * self._pi.action_scale
            a += clipped_noise
        a = a.clamp(self.act_spec_low, self.act_spec_high)
        return a

    @property
    def exploration_noise(self) -> h.LinearSchedule:
        return self._exploration_noise_schedule()

    def metrics(self, batch: ReplayBufferSamples) -> dict:
        metrics = self.encoder.metrics(batch)

        metrics.update({"enc": h.calc_mean_opt_moments(self.enc_opt)})
        metrics.update({"Q": h.calc_mean_opt_moments(self.q_opt)})
        metrics.update({"pi": h.calc_mean_opt_moments(self.pi_opt)})

        # TODO add dormant neuron ratio stuff
        # metrics.update(h.calc_dormant_neuron_ratio(batch, agent=self))

        return metrics

    @property
    def total_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
