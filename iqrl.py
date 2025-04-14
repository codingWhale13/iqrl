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
from einops import einsum, rearrange
from tensordict import LazyStackedTensorDict, TensorDict, TensorDictBase
from torchrl.data import Bounded, CompositeSpec
from utils import ReplayBuffer, ReplayBufferSamples


logger = logging.getLogger(__name__)

# SAC constants
LOG_STD_MAX = 2
LOG_STD_MIN = -5


@dataclass
class iQRLConfig:
    """Config for iQRL"""

    """Map environment states to latent states before using them in other components"""
    use_obs_encoder: bool = True  # Used in original iQRL, thus defaults to True
    """Map policy actions to latent actions before using them in dynamics and critic"""
    use_action_encoder: bool = False  # Not in original iQRL, thus defaults to False
    """Condition {encoders, transition dynamics, actor, critic} on body & task IDs"""
    condition_encoders: bool = True
    condition_dynamics: bool = True
    condition_actor: bool = True
    condition_critic: bool = True
    condition_reward: bool = True
    """MLP dims for actor/critic/dynamics"""
    mlp_dims: List[int] = field(default_factory=lambda: [512, 512])
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
    """In critic update, next_s can be "encoded" (as in iQRL) or "rollout\""""
    critic_next_s: str = "encoded"
    """Body and task embedding size; use None for one-hot encoding instead"""
    context_dim: Optional[int] = None  # Sensible default for embedding size: 96
    """What observation types to use? ["state"] or ["pixels"] or ["state", "pixels"]"""
    obs_types: List[str] = field(default_factory=lambda: ["state"])
    """Which model-free RL algorithm to use, TD3 or SAC"""
    rl_algo: str = "TD3"
    """Use N-step returns for Q-learning? Set to -1 for lambda-returns"""
    nstep: int = 1
    median_lambda_return: bool = True  # Used only for lambda-returns, i.e. nstep=-1
    return_lambda: float = 0.95  # Used only if nstep=-1 and median_lambda_return=False

    """SAC CONFIG"""
    """Entropy regularization coefficient"""
    sac_alpha: float = 0.2
    """The learning rate of the Q network network optimizer"""
    sac_lr: float = 1e-3
    """Automatic tuning of the entropy coefficient"""
    sac_autotune: bool = True

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
    """Learning rate for encoder/dynamics/reward"""
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
    """Reward coefficient"""
    reward_coef: float = 1.0
    """Consistency coefficient"""
    consistency_coef: float = 1.0
    """Use "mse" or "soft-ce" (soft cross-entropy) in critic and reward updates"""
    Q_and_rew_loss: str = "mse"
    """Number of bins and value range (only used if Q_and_rew_loss="soft-ce")"""
    num_bins: int = 101
    vmin: float = -10.0
    vmax: float = 10.0
    bin_size: float = -1  # Depends on num_bins, vmin, vmax; determined later

    """Which loss function to use for consistency loss?"""
    consistency_loss: str = "cosine"  # "cross-entropy", "mse", "cosine"
    """Predict logits with dynamics NN or use cosine/mse between pred and codebook?  (only for cross-entropy)"""
    ce_logits_mode: str = "standard"  # "standard", cosine", "mse"
    """How to get propagate the state dist. during training (only for cross-entropy)"""
    unc_prop_mode: str = "sample"  # Literal["sample", "sample-no-grad", "weighted-avg"]
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
    """Logging frequency, only takes effect if log_during_update=True"""
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

        out_dim = act_dim if cfg.rl_algo == "TD3" else act_dim * 2  # SAC -> 2 heads
        self.mlp = h.mlp(in_dim, self.cfg.mlp_dims, out_dim)

    def forward(
        self, z: torch.Tensor, ctx: list[torch.Tensor]
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        x = torch.cat(ctx + [z], -1) if self.cfg.condition_actor else z
        if self.cfg.rl_algo == "TD3":
            a = self.mlp(x)
            a = torch.tanh(a)
            a = a * self.action_scale + self.action_bias

            return a, None, None
        else:  # SAC
            mean, log_std = self.mlp(x).chunk(2, dim=-1)
            log_std = torch.tanh(log_std)
            log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)
            std = log_std.exp()

            normal = torch.distributions.Normal(mean, std)
            x_t = normal.rsample()  # For reparameterization trick (mean + std * N(0,1))
            y_t = torch.tanh(x_t)
            action = y_t * self.action_scale + self.action_bias
            log_prob = normal.log_prob(x_t)

            # Enforcing action bound
            log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
            log_prob = log_prob.sum(1, keepdim=True)
            mean = torch.tanh(mean) * self.action_scale + self.action_bias

            return action, log_prob, mean


class Critic(nn.Module):
    def __init__(self, cfg: iQRLConfig, in_dim: int):
        super().__init__()
        self.cfg = cfg

        qs = [
            h.mlp(
                in_dim=in_dim,
                mlp_dims=cfg.mlp_dims,
                out_dim=1 if self.cfg.Q_and_rew_loss == "mse" else cfg.num_bins,
                dropout=cfg.q_dropout,
            ).to(cfg.device)
            for _ in range(cfg.num_critics)
        ]
        for q in qs:
            h.orthogonal_init(q.parameters())

        self.qs = h.Ensemble(qs)

    def forward(
        self,
        z: torch.Tensor,
        a: torch.Tensor,
        ctx: list[torch.Tensor],
        return_type: str = "all",
    ):
        x = torch.cat(ctx + [z, a] if self.cfg.condition_critic else [z, a], -1)
        qs = self.qs(x)
        if return_type == "all":
            return qs

        # Sample two Q values
        if self.cfg.q_sample_size is not None:
            idxs = torch.randperm(qs.shape[0])[: self.cfg.q_sample_size]
            qs = qs[idxs]

        # Map back Q-values: last dim goes from bin_size to 1
        if self.cfg.Q_and_rew_loss == "soft-ce":
            qs = h.two_hot_inv(qs, self.cfg)

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
        ctx_dim: int,
        org_latent_dim: int,
        n_body: int,
        n_task: int,
    ):
        super().__init__()
        self.cfg = cfg
        self.obs_dim = obs_dim

        ##### Prepare for adding body and task context (either 1-hot or embedded) #####
        self.n_body = n_body
        self.n_task = n_task
        if cfg.context_dim is not None:
            self._body_emb = nn.Embedding(
                self.n_body, cfg.context_dim, max_norm=1, device=self.cfg.device
            )
            self._task_emb = nn.Embedding(
                self.n_task, cfg.context_dim, max_norm=1, device=self.cfg.device
            )

        ##### Configure FSQ stuff #####
        self.org_latent_dim = org_latent_dim
        self.num_channels = len(cfg.fsq_levels)
        if cfg.use_fsq:
            self._fsq = h.FSQ(levels=cfg.fsq_levels)

        ##### Init encoders #####
        self._encoder = nn.ModuleDict()
        if "state" in cfg.obs_types and cfg.use_obs_encoder:
            self._encoder["state"] = h.mlp(
                in_dim=obs_dim + (ctx_dim if cfg.condition_encoders else 0),
                mlp_dims=cfg.enc_mlp_dims,
                out_dim=latent_obs_dim,
                dropout=cfg.enc_dropout,
            )
        if cfg.use_action_encoder:
            self._encoder["action"] = h.mlp(
                in_dim=act_dim + (ctx_dim if cfg.condition_encoders else 0),
                mlp_dims=cfg.enc_mlp_dims,
                out_dim=latent_act_dim,
                dropout=cfg.enc_dropout,
            )
        if cfg.use_tar_enc:
            self._encoder_tar = copy.deepcopy(self._encoder).requires_grad_(False)

        ##### Init transition dynamics #####
        trans_out_dim = self.cfg.latent_dim
        if self.cfg.consistency_loss == "cross-entropy":
            if self.cfg.ce_logits_mode == "standard":
                """If training dynamics w/ cross entropy change output dim"""
                assert cfg.use_fsq
                trans_out_dim = int(self.org_latent_dim * self._fsq.codebook_size)
        self._trans = h.mlp(
            in_dim=latent_obs_dim
            + latent_act_dim
            + (ctx_dim if cfg.condition_dynamics else 0),
            mlp_dims=cfg.mlp_dims,
            out_dim=trans_out_dim,
        )

        ##### Init optional reward model #####
        if cfg.use_rew_loss:
            self._reward = h.mlp(
                in_dim=latent_obs_dim
                + latent_act_dim
                + (ctx_dim if cfg.condition_reward else 0),
                mlp_dims=cfg.mlp_dims,
                out_dim=1 if cfg.Q_and_rew_loss == "mse" else cfg.num_bins,
            )

    def get_context(self, obs: TensorDictBase) -> list[torch.Tensor]:
        """
        Returns body and task representation, if available.
        The representations will be one-hot if context_dim is None, embeddings otherwise.
        """

        context = []
        body_id = obs.get("body_id")
        task_id = obs.get("task_id")
        if body_id is not None:
            body_id = body_id.long().squeeze(-1).to(self.cfg.device)
        if task_id is not None:
            task_id = task_id.long().squeeze(-1).to(self.cfg.device)

        if self.cfg.context_dim is None:
            # Use one-hot encoding
            if body_id is not None:
                body = nn.functional.one_hot(body_id, self.n_body).to(self.cfg.device)
                context.append(body)
            if task_id is not None:
                task = nn.functional.one_hot(task_id, self.n_task).to(self.cfg.device)
                context.append(task)
        else:
            # Use embedding
            if body_id is not None:
                context.append(self._body_emb(body_id).to(self.cfg.device))
            if task_id is not None:
                context.append(self._task_emb(task_id).to(self.cfg.device))

        return context

    def encode_obs(self, obs: TensorDictBase, tar: bool = False) -> TensorDictBase:
        if not self.cfg.use_obs_encoder:
            return obs  # Identity mapping

        if isinstance(obs, LazyStackedTensorDict):
            obs_tensor = obs.get_nestedtensor("state").to_padded_tensor(padding=0.0)
        else:
            obs_tensor = obs["state"]
        p1d = (0, self.obs_dim - obs_tensor.shape[-1])  # Don't assume inherent max obs
        obs_padded = F.pad(obs_tensor, p1d, "constant", 0.0).to(self.cfg.device)

        if self.cfg.condition_encoders:
            obs_padded = torch.cat(self.get_context(obs) + [obs_padded], dim=-1)

        if tar:
            z = self._encoder_tar["state"](obs_padded)
        else:
            z = self._encoder["state"](obs_padded)
        td = TensorDict({"state": z}, batch_size=obs.batch_size)

        if self.cfg.use_fsq:
            td.update(self.quantize(z))
        else:
            td.update({"codes": z})

        return td

    def encode_action(
        self, action: torch.Tensor, ctx: list[torch.Tensor], tar: bool = False
    ) -> torch.Tensor:
        if not self.cfg.use_action_encoder:
            return action  # Identity mapping

        # NOTE: No padding required because action comes from padded replay buffer
        if self.cfg.condition_encoders:
            action = torch.cat(ctx + [action], dim=-1)

        if tar:
            za = self._encoder_tar["action"](action)
        else:
            za = self._encoder["action"](action)

        return za  # NOTE: actions are not being quantized currently

    def trans(
        self,
        z: torch.Tensor,
        a: torch.Tensor,
        ctx: list[torch.Tensor],
        unc_prop_mode: Optional[str] = None,
    ):
        za = torch.concat((ctx + [z, a] if self.cfg.condition_dynamics else [z, a]), -1)

        if (
            self.cfg.consistency_loss == "cross-entropy"
            and self.cfg.ce_logits_mode == "standard"
        ):
            """Make predictions with dynamics as NN classifier"""
            # Returns logits for each class
            logits = self._trans(za)
            logits = logits.reshape(-1, self.org_latent_dim, self._fsq.codebook_size)

            if unc_prop_mode is None:
                unc_prop_mode = self.cfg.unc_prop_mode

            # Convert latent state logits to an actual latent state
            if "sample-no-grad" in unc_prop_mode:

                def gumbel_sample(logits):
                    gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits)))
                    adjusted_logits = logits + gumbel_noise
                    return torch.argmax(adjusted_logits, dim=-1)

                indices = gumbel_sample(logits)
                next_z = self._fsq.implicit_codebook[indices].flatten(-2)
                next_z_dict = {
                    "codes": next_z,
                    "logits": logits,
                    "indices": indices.to(torch.float),
                }
            elif "sample" in unc_prop_mode:
                z_one_hot = torch.nn.functional.gumbel_softmax(
                    logits, tau=1, hard=True, dim=-1
                )
                codebook = self._fsq.implicit_codebook
                next_z = einsum(z_one_hot, codebook, "b d c, c l -> b d l")
                next_z = rearrange(next_z, "b d l -> b (d l)")
                next_z_dict = {
                    "codes": next_z,
                    "logits": logits,
                    "one-hot": z_one_hot.flatten(-2),
                }
            elif "weighted-avg" in unc_prop_mode:
                probs = F.softmax(logits, dim=-1)
                codebook = self._fsq.implicit_codebook
                next_z = einsum(probs, codebook, "b d c, c l -> b d l")
                next_z = rearrange(next_z, "b d l -> b (d l)")
                next_z_dict = {"codes": next_z, "logits": logits}
            else:
                raise NotImplementedError
        else:
            """Make predictions with dynamics regression model"""
            delta_z = self._trans(za)
            next_z = z + delta_z if self.cfg.use_delta else delta_z
            if self.cfg.use_fsq:
                next_z = self.quantize(next_z)["codes"]

            next_z_dict = {"codes": next_z}

        if self.cfg.use_fsq:
            shape = *next_z.shape[0:-1], self.org_latent_dim, self.num_channels
        else:
            shape = *next_z.shape[0:-1], self.org_latent_dim
        next_z_dict.update({"z": next_z.reshape(shape)})

        return TensorDict(
            next_z_dict,
            batch_size=torch.Size([z.shape[0]]),
            device=self.cfg.device,
        )

    def reward(
        self, z: torch.Tensor, a: torch.Tensor, ctx: list[torch.Tensor]
    ) -> torch.Tensor:
        za = torch.cat(ctx + [z, a] if self.cfg.condition_reward else [z, a], -1)
        r = self._reward(za)
        return r

    def quantize(self, z: torch.Tensor) -> dict[str, torch.Tensor]:
        """Quantize the latent state"""
        td = self._fsq(z)
        td["state"] = td["codes"]
        return td

    def latent_rollout(self, batch: ReplayBufferSamples, grad: bool) -> TensorDict:
        """
        Using only the initial obs, predict future latent states with learned dynamics.
        NOTE: Returns H+1 latent states, because zs[0] is just the latent initial state.
        NOTE: latent rollout is just used for consistency loss - MultiQRL is model-free.
        """
        # Prepare zs TensorDict
        zs = {
            "codes": torch.empty(
                self.cfg.horizon + 1,
                self.cfg.batch_size,
                self.cfg.latent_dim,
                device=self.cfg.device,
            )
        }
        if self.cfg.consistency_loss == "cross-entropy":
            zs.update(
                {
                    "logits": torch.empty(
                        self.cfg.horizon + 1,
                        self.cfg.batch_size,
                        self.org_latent_dim,
                        self._fsq.codebook_size,
                        device=self.cfg.device,
                    )
                }
            )
        zs = TensorDict(
            zs,
            batch_size=torch.Size([self.cfg.horizon + 1, self.cfg.batch_size]),
            device=self.cfg.device,
        )

        with torch.set_grad_enabled(grad):
            # Prepare context of correct dimensionality and encode all actions
            ctx = self.get_context(batch.observations)
            ctx_t = self.get_context(batch.observations[0])
            actions = self.encode_action(batch.actions, ctx=ctx, tar=False)

            # Rollout next H latent states
            z = self.encode_obs(batch.observations[0])["codes"]
            zs["codes"][0] = z
            dones = torch.zeros_like(batch.dones[0], dtype=torch.bool)
            terminateds_or_dones = torch.zeros_like(batch.dones, dtype=torch.bool)
            for t in range(self.cfg.horizon):
                dones = torch.where(terminateds_or_dones[t], dones, batch.dones[t])
                terminateds_or_dones[t] = torch.logical_or(
                    terminateds_or_dones[t],
                    torch.logical_or(dones, batch.terminateds[t]),
                )

                # Predict next latent state
                next_z = self.trans(z=z, a=actions[t], ctx=ctx_t)
                zs[t + 1] = next_z

                # Don't forget this
                z = next_z["codes"]

        zs["state"] = zs["codes"]  # For more convenient access

        return zs

    def loss(self, batch: ReplayBufferSamples) -> Tuple[torch.Tensor, dict]:
        tc_loss = torch.zeros(1).to(self.cfg.device)
        reward_loss = torch.zeros(1).to(self.cfg.device)

        ##### Create targets #####
        with torch.no_grad():
            zs_tar = self.encode_obs(batch.next_observations, tar=True)

        ##### Latent rollout #####
        zs = self.latent_rollout(batch, grad=True)

        rho = torch.tensor([self.cfg.rho**t for t in range(self.cfg.horizon)]).to(
            self.cfg.device
        )
        dones = batch.dones.to(torch.int)

        ##### (Optional) Reward prediction loss #####
        if self.cfg.use_rew_loss:
            # Reward target
            r_tar = batch.rewards

            # Reward prediction
            ctx = self.get_context(batch.observations)
            actions = self.encode_action(batch.actions, ctx=ctx, tar=False)
            r_pred = self.reward(z=zs["codes"][:-1], a=actions, ctx=ctx).squeeze(-1)

            if self.cfg.Q_and_rew_loss == "mse":
                assert r_pred.ndim == r_tar.ndim == 2
                _reward_loss = (r_pred - r_tar) ** 2
            elif self.cfg.Q_and_rew_loss == "soft-ce":
                _reward_loss = torch.empty_like(r_tar)
                for t in range(self.cfg.horizon):
                    _reward_loss[t] = h.soft_ce(r_pred[t], r_tar[t], self.cfg)
            _rho_reward_loss = rho * torch.mean((1 - dones) * _reward_loss, -1)
            reward_loss = torch.mean(_rho_reward_loss)

        ##### Temporal consistency loss #####
        if self.cfg.use_tc_loss:
            if self.cfg.consistency_loss == "cross-entropy":
                """Cross entropy"""
                if self.cfg.ce_logits_mode in ["cosine", "mse"]:
                    """If not predicting logits with dynamics NN use alternative method"""
                    zs_ = zs["codes"][1:].view(
                        self.cfg.horizon,
                        self.cfg.batch_size,
                        int(self.cfg.latent_dim / self.num_channels),
                        self.num_channels,
                    )[..., None, :]
                    codebook = self._fsq.implicit_codebook[None, None, None, ...]
                    if self.cfg.ce_logits_mode == "cosine":
                        """Cosine similarity with codebook"""
                        # TODO use compute_logits like CLIP
                        zs["logits"][1:] = nn.CosineSimilarity(dim=-1, eps=1e-6)(
                            zs_, codebook
                        )
                    elif self.cfg.ce_logits_mode == "mse":
                        """Inner product with codebook"""
                        zs["logits"][1:] = torch.einsum(
                            "hbdic,hbdCc->hbdC", zs_, codebook
                        )
                _tc_loss = torch.vmap(torch.vmap(F.cross_entropy))(
                    zs["logits"][1:],
                    zs_tar["indices"].to(torch.long),
                )
            elif self.cfg.consistency_loss == "cosine":
                """Cosine similarity"""
                _tc_loss = -nn.CosineSimilarity(dim=-1, eps=1e-6)(
                    zs["codes"][1:], zs_tar["codes"]
                )
            elif self.cfg.consistency_loss == "mse":
                """Mean squared error"""
                _tc_loss = torch.mean((zs["codes"][1:] - zs_tar["codes"]) ** 2, dim=-1)
            else:
                raise NotImplementedError(
                    f"cfg.consistency_loss should be 'cross-entropy', 'mse', 'cosine', not {self.cfg.consistency_loss}"
                )

            _rho_tc_loss = rho * torch.mean((1 - dones) * _tc_loss, -1)
            tc_loss = torch.mean(_rho_tc_loss)

        loss = self.cfg.consistency_coef * tc_loss + self.cfg.reward_coef * reward_loss
        info = {
            "tc_loss": tc_loss.item(),
            "reward_loss": reward_loss.item(),
            "enc_loss": loss.item(),
            "z_min": torch.min(zs["codes"]).item(),
            "z_max": torch.max(zs["codes"]).item(),
            "z_mean": torch.mean(zs["codes"].to(torch.float)).item(),
            "z_median": torch.median(zs["codes"]).item(),
        }
        if self.cfg.use_rew_loss:
            info.update(
                {
                    "r_min": r_pred.min().item(),
                    "r_max": r_pred.max().item(),
                    "r_mean": r_pred.mean().item(),
                }
            )

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

        return metrics


class iQRL(nn.Module):
    def __init__(
        self,
        cfg: iQRLConfig,
        obs_specs: list[CompositeSpec],
        act_specs: list[Bounded],
        n_body: int,
        n_task: int,
        ids_to_dims: dict[tuple[int, int], tuple[int, int]],
    ):
        super().__init__()
        self.cfg = cfg
        self.ids_to_dims = ids_to_dims
        self.use_td_lambda = self.cfg.nstep == -1

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
        org_latent_dim = copy.copy(cfg.latent_dim)  # Original latent dim
        if cfg.use_fsq:
            assert cfg.use_obs_encoder, "Can only use FSQ when using an obs encoder"
            self.num_channels = len(cfg.fsq_levels)
            if not cfg.latent_dim % self.num_channels == 0:
                raise NotImplementedError(
                    "latent_dim must be divisible by number of FSQ channels"
                )
            if self.num_channels > 1:
                logger.info(
                    f"Increasing latent dim from {cfg.latent_dim} to "
                    f"{cfg.latent_dim*self.num_channels} to account for FSQ channels"
                )
                cfg.latent_dim *= self.num_channels

        ##### Prepare for twohot loss: Determine bin size for discrete regression #####
        self.cfg.bin_size = (cfg.vmax - cfg.vmin) / (cfg.num_bins - 1)

        ##### Calculate max dims of observations, actions and body&task IDs #####
        obs_dim = max(np.prod(obs["state"].shape).item() for obs in obs_specs)
        self.act_dims = [np.prod(act_spec.shape).item() for act_spec in act_specs]
        act_dim = max(self.act_dims)
        keys = [c for c in ["body_id", "task_id"] if c in obs_specs[0].keys()]
        if cfg.context_dim is None:  # IDs will be one-hot encoded
            ctx_dim = n_body * ("body_id" in keys) + n_task * ("task_id" in keys)
        else:  # IDs will be embedded
            ctx_dim = cfg.context_dim * len(keys)

        ##### Calculate dimensions of (optional) latent spaces #####
        latent_obs_dim = cfg.latent_dim if cfg.use_obs_encoder else obs_dim
        latent_act_dim = act_dim
        if cfg.use_action_encoder:
            latent_act_dim *= cfg.latent_action_dim_factor

        ##### Init encoders, dynamics, and optionally reward model #####
        self.encoder = Encoder(
            cfg,
            obs_dim=obs_dim,
            act_dim=act_dim,
            latent_obs_dim=latent_obs_dim,
            latent_act_dim=latent_act_dim,
            ctx_dim=ctx_dim,
            org_latent_dim=org_latent_dim,
            n_body=n_body,
            n_task=n_task,
        ).to(cfg.device)
        if cfg.compile:
            self.encoder = torch.compile(self.encoder, mode="default")
        self.enc_opt = torch.optim.AdamW(self.encoder.parameters(), lr=cfg.enc_lr)

        ##### Init actor network and its target network #####
        self._pi = Actor(
            cfg,
            in_dim=latent_obs_dim + (ctx_dim if cfg.condition_actor else 0),
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
            + (ctx_dim if cfg.condition_critic else 0),
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

        ##### Automatic entropy tuning #####
        if cfg.rl_algo == "SAC" and cfg.sac_autotune:
            self.target_entropy = -act_dim
            self.sac_log_alpha = torch.zeros(1, requires_grad=True, device=cfg.device)
            self.sac_alpha = self.sac_log_alpha.exp().item()
            self.sac_alpha_optimizer = torch.optim.Adam(
                [self.sac_log_alpha], lr=cfg.sac_lr
            )
        else:
            self.sac_alpha = cfg.sac_alpha

        # Counters for number of param updates
        self.critic_update_counter = 0
        self.pi_update_counter = 0

        self.state_norm = None  # If cfg.normalize_states==True, will be set later

    def update(
        self,
        replay_buffer: ReplayBuffer,
        num_new_transitions: int,
        fake: bool = False,
        rb_idx: Optional[int] = None,
    ) -> dict:
        """Update representation and RL at same time (if fake=True, only return info)"""
        num_updates = int(num_new_transitions * self.cfg.utd_ratio)
        info = {}

        if self.cfg.verbose and not fake:
            logger.info(f"Performing {num_updates} iQRL updates...")
        for i in range(num_updates):
            batch = replay_buffer.sample(rb_idx=rb_idx)

            # Update enc less frequently than actor/critic
            if i % self.cfg.enc_update_freq == 0:
                info.update(self.representation_update_step(batch=batch, fake=fake))

            # Map observations to latent states
            with torch.no_grad():
                if self.cfg.critic_next_s == "encoded":
                    z = self.encoder.encode_obs(batch.observations, tar=False)
                    next_z = self.encoder.encode_obs(batch.next_observations, tar=False)
                elif self.cfg.critic_next_s == "rollout":
                    # Do rollout, now *after* representation update
                    zs = self.encoder.latent_rollout(batch, grad=False)
                    z = zs[:-1]
                    next_z = zs[1:]
            batch = batch._replace(z=z, next_z=next_z)

            # Avoid edge case when making nstep batch (H, B, whatever) -> (B, whatever)
            if self.cfg.horizon == 1:
                raise NotImplementedError("Check N-step batch is made correctly if h=1")

            ##### Update critic #####
            info.update(self.critic_update_step(batch=batch, fake=fake))

            ##### Update actor less frequently than critic #####
            if self.critic_update_counter % self.cfg.actor_update_freq == 0:
                info.update(self.pi_update_step(batch=batch, fake=fake))

            if i % self.cfg.logging_freq == 0 and not fake:
                if self.cfg.verbose:
                    logger.info(
                        f"Iteration {i} | loss {info['enc_loss']:.3} | tc loss {info['tc_loss']:.3} | reward loss {info['reward_loss']:.3}"
                    )
                if wandb.run is not None and self.cfg.log_during_update:
                    wandb.log(info)

        if not fake:
            # Update exploration noise
            info["exploration_noise"] = self.exploration_noise
            if wandb.run is not None and self.cfg.log_during_update:
                wandb.log({"exploration_noise": self.exploration_noise})
            self._exploration_noise_schedule.step()

            if self.cfg.verbose:
                logger.info("Finished training iQRL")

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

        self.encoder.eval()
        return info

    def get_nstep_return(self, nstep_batch: ReplayBufferSamples) -> torch.Tensor:
        with torch.no_grad():
            next_z = nstep_batch.next_z["state"]
            # Calculate next_a, i.e. the action that agent takes in next_z
            ctx = self.encoder.get_context(nstep_batch.observations)
            next_a_raw, _, _ = self.pi(
                next_z, ctx=ctx, tar=True, eval_mode=True, smooth=True
            )
            next_a_raw *= nstep_batch.observations["act_mask"]
            next_a = self.encoder.encode_action(next_a_raw, ctx=ctx)

            # Calculate Q target, using next_s and next_a to "peek into the future"
            min_q_next_tar = self.Q_tar(
                z=next_z, a=next_a, ctx=ctx, return_type="min"
            ).squeeze(-1)
            assert min_q_next_tar.shape == nstep_batch.rewards.shape

            nstep_return = (
                nstep_batch.rewards
                + (1 - nstep_batch.terminateds)
                * nstep_batch.next_state_gammas
                * min_q_next_tar
            )

        return nstep_return

    def _get_lambda_return(
        self, nstep_returns: torch.Tensor, lam: float
    ) -> torch.Tensor:
        lambda_return = torch.zeros(self.cfg.batch_size, device=self.cfg.device)
        for n in range(1, self.cfg.horizon + 1):
            # NOTE: nstep_returns use 0-based index -> nstep_returns[n - 1]
            weighted_nstep_return = lam ** (n - 1) * nstep_returns[n - 1]
            if n < self.cfg.horizon:
                weighted_nstep_return *= 1 - lam  # Weight by normalization constant
            lambda_return += weighted_nstep_return

        return lambda_return

    def get_lambda_return(self, all_nstep_batch: ReplayBufferSamples) -> torch.Tensor:
        nstep_returns = self.get_nstep_return(all_nstep_batch)

        if self.cfg.median_lambda_return:
            # Calculate median lambda-return of k+1 lambda values, same as Daley (2018)
            k = 20
            lambda_returns_list = [
                self._get_lambda_return(nstep_returns, lam=i / k) for i in range(k + 1)
            ]
            return torch.stack(lambda_returns_list, dim=0).median(dim=0).values
        else:
            return self._get_lambda_return(nstep_returns, lam=self.cfg.return_lambda)

    def critic_update_step(
        self, batch: ReplayBufferSamples, fake: bool = False
    ) -> dict:
        self.Q.train()
        self.Q_tar.train()

        # Check batch shapes
        assert batch.rewards.shape[0] == batch.observations.shape[0]
        assert batch.z is not None
        assert batch.rewards.shape == (self.cfg.horizon, self.cfg.batch_size)

        # Extract current (z, a) from full batch to make Q-prediction
        z = batch.z["state"][0]
        ctx = self.encoder.get_context(batch.observations[0])
        a = self.encoder.encode_action(batch.actions[0], ctx=ctx)
        q_values = self.Q(z=z, a=a, ctx=ctx, return_type="all").squeeze(-1)

        # Make Q-target
        with torch.no_grad():
            if self.use_td_lambda:
                all_nstep_batch = utils.to_all_nstep(
                    batch, self.cfg.horizon, self.cfg.gamma
                )
                next_q_value = self.get_lambda_return(all_nstep_batch)
            else:
                nstep_batch = utils.to_nstep(
                    batch, nstep=self.cfg.nstep, gamma=self.cfg.gamma
                )
                next_q_value = self.get_nstep_return(nstep_batch)

        # Calculate Q-loss
        if self.cfg.Q_and_rew_loss == "mse":
            next_q_value = next_q_value.broadcast_to(q_values.shape)  # For num_critics
            q_loss = F.mse_loss(q_values, next_q_value)
        elif self.cfg.Q_and_rew_loss == "soft-ce":
            q_loss = 0
            for i in range(self.cfg.num_critics):
                q_loss += h.soft_ce(q_values[i], next_q_value, self.cfg).mean()
            q_loss /= self.cfg.num_critics

        if not fake:  # Actually perform the optimization step
            self.critic_update_counter += 1

            ##### Optimize critic #####
            self.q_opt.zero_grad(set_to_none=True)
            q_loss.backward()
            self.q_opt.step()

            ##### Update the target network #####
            h.soft_update_params(self.Q, self.Q_tar, tau=self.cfg.tau)

        # For logging, change two-hot encoded vectors back to scalars
        q_values = h.two_hot_inv(q_values, self.cfg)

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

        # Recover all H+1 zs from z and next_z -> do (H+1)*B updates instead of H*B
        z = torch.cat([batch.z["state"], batch.next_z["state"][-1].unsqueeze(0)])

        # Get policy actions
        H = self.cfg.horizon
        ctx = self.encoder.get_context(batch.observations[0])
        for i in range(len(ctx)):
            ctx[i] = ctx[i].expand(H + 1, -1, -1)
        pi_actions, log_pi, _ = self.pi(z=z, ctx=ctx, eval_mode=True)
        pi_actions *= batch.observations["act_mask"][0].expand(H + 1, -1, -1)
        pi_actions = self.encoder.encode_action(pi_actions, ctx=ctx)

        if self.cfg.use_offline_data:
            Q_values = self.Q(z=z, a=pi_actions, ctx=ctx, return_type="avg")
            # Add behavior cloning regularization
            lmbda = self.cfg.bc_alpha / Q_values.abs().mean().detach()
            pi_loss = -lmbda * Q_values.mean() + F.mse_loss(pi_actions, batch.actions)
        elif self.cfg.rl_algo == "TD3":
            Q_values = self.Q(z=z, a=pi_actions, ctx=ctx, return_type="avg")
            # Discount future predicitons by rho to take into account higher uncertainty
            rho = torch.tensor([self.cfg.rho**t for t in range(H + 1)])
            pi_loss = -(rho.to(self.cfg.device) * Q_values.mean(dim=(1, 2))).mean()
        elif self.cfg.rl_algo == "SAC":
            Q_values = self.Q(z=z, a=pi_actions, ctx=ctx, return_type="min")
            pi_loss = (self.cfg.sac_alpha * log_pi - Q_values).mean()

        if not fake:  # Actually perform the optimization step
            ##### Optimize actor #####
            self.pi_opt.zero_grad(set_to_none=True)
            pi_loss.backward()
            self.pi_opt.step()

            ##### Update the target network #####
            h.soft_update_params(self._pi, self._pi_tar, tau=self.cfg.tau)

        self._pi.eval()

        if self.cfg.rl_algo == "SAC" and self.cfg.sac_autotune:
            with torch.no_grad():
                _, log_pi, _ = self._pi(z=z, ctx=ctx)
            alpha_loss = (
                -self.sac_log_alpha.exp() * (log_pi + self.target_entropy).detach()
            ).mean()

            self.sac_alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.sac_alpha_optimizer.step()
            self.sac_alpha = self.sac_log_alpha.exp().item()

        info = {
            "actor_loss": pi_loss.item(),
            "actor_update_counter": self.pi_update_counter,
        }
        if self.cfg.rl_algo == "SAC":
            info["alpha"] = self.sac_alpha
            if self.cfg.sac_autotune:
                info["alpha_loss"] = alpha_loss.item()

        return info

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
        ctx = self.encoder.get_context(obs)
        a, _, _ = self.pi(s["state"], ctx, tar=False, eval_mode=eval_mode)

        if is_flat_obs:
            body_id = obs["body_id"][0].item()
            task_id = obs["task_id"][0].item()
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
        z: torch.Tensor,
        ctx: list[torch.Tensor],
        tar: bool = False,
        eval_mode: bool = False,
        smooth: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        a, log_prob, mean = self._pi_tar(z, ctx) if tar else self._pi(z, ctx)
        if not eval_mode:
            a += torch.normal(0, self._pi.action_scale * self.exploration_noise)
        if smooth:
            clipped_noise = (
                torch.randn_like(a, device=self.cfg.device) * self.cfg.policy_noise
            ).clamp(-self.cfg.noise_clip, self.cfg.noise_clip) * self._pi.action_scale
            a += clipped_noise
        a = a.clamp(self.act_spec_low, self.act_spec_high)
        return a, log_prob, mean

    @property
    def exploration_noise(self) -> h.LinearSchedule:
        return self._exploration_noise_schedule()

    def metrics(self, batch: ReplayBufferSamples) -> dict:
        metrics = self.encoder.metrics(batch)

        metrics.update({"enc": h.calc_mean_opt_moments(self.enc_opt)})
        metrics.update({"Q": h.calc_mean_opt_moments(self.q_opt)})
        metrics.update({"pi": h.calc_mean_opt_moments(self.pi_opt)})

        return metrics

    @property
    def total_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
