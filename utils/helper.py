#!/usr/bin/env python3
import copy
from typing import List, Optional, Sequence, Union

import numpy as np
import torch
import torch.nn as nn
from torch.func import functional_call, stack_module_state
from torch.linalg import cond, matrix_rank
from vector_quantize_pytorch import FSQ as _FSQ
from tensordict import LazyStackedTensorDict


def soft_update_params(model, model_target, tau: float):
    """Update slow-moving average of online network (target network) at rate tau."""
    with torch.no_grad():
        for params, params_target in zip(model.parameters(), model_target.parameters()):
            params_target.data.lerp_(params.data, tau)
            # One below is from CleanRL
            # params_target.data.copy_(tau * params.data + (1 - tau) * params_target.data)


class ContextSequential(nn.Sequential):
    def forward(self, x: torch.Tensor, ctx: list[torch.Tensor]):
        for module in self:
            x = module(x, ctx)
        return x


def mlp(
    in_dim: int,
    mlp_dims: Union[int, list[int]],
    out_dim: int,
    ctx_dim: int = 0,
    condition_layer: Optional[str] = None,  # None or "first" or "all"
    condition_ln: bool = False,
    act_fn=None,
    dropout=0.0,
    norm_mode: str = "ln",
    norm_after_act: bool = False,
):
    """
    MLP with LayerNorm, Mish activations, and optionally dropout.

    Adapted from https://github.com/tdmpc2/tdmpc2-eval/blob/main/helper.py

    If both ctx_dim and condition are specified:
    - in_dim increases by ctx_dim, if condition in ["first", "all"]
    - all mlp_dims increase by ctx_dim, if condition == "all"
    """
    if isinstance(mlp_dims, int):
        mlp_dims = [mlp_dims]

    dims = [int(in_dim)] + mlp_dims + [int(out_dim)]
    mlp = nn.ModuleList()
    mlp.append(
        NormedLinear(
            dims[0],
            dims[1],
            ctx_dim=ctx_dim if condition_layer in ["first", "all"] else None,
            condition_ln=condition_ln,
            dropout=dropout,
            norm_mode=norm_mode,
            norm_after_act=norm_after_act,
        )
    )

    add_to_next_in_dim = 0  # Conditioning LayerNorm changes outgoing dims of layers
    if condition_layer in ["first", "all"] and condition_ln:
        add_to_next_in_dim = ctx_dim

    for i in range(1, len(dims) - 2):
        mlp.append(
            NormedLinear(
                dims[i] + add_to_next_in_dim,
                dims[i + 1],
                ctx_dim=ctx_dim if condition_layer == "all" else None,
                condition_ln=condition_ln,
                norm_mode=norm_mode,
                norm_after_act=norm_after_act,
            )
        )
        if condition_layer == "all" and condition_ln:
            add_to_next_in_dim = ctx_dim
        else:
            add_to_next_in_dim = 0

    mlp.append(
        NormedLinear(
            dims[-2] + add_to_next_in_dim,
            dims[-1],
            ctx_dim=ctx_dim if condition_layer == "all" else None,
            condition_ln=condition_ln,
            act=act_fn,
            norm_mode=norm_mode,
            norm_after_act=norm_after_act,
        )
        if act_fn
        else ContextLinear(
            in_features=dims[-2] + add_to_next_in_dim,
            out_features=dims[-1],
            ctx_dim=ctx_dim if condition_layer == "all" else None,
        )
    )
    return ContextSequential(*mlp)


class FSQ(_FSQ):
    """
    Finite Scalar Quantization
    """

    def __init__(self, levels: List[int]):
        super().__init__(levels=levels)
        self.levels = levels
        self.num_channels = len(levels)

    def forward(self, z):
        shp = z.shape
        z = z.view(*shp[:-1], -1, self.num_channels)
        if z.ndim > 3:  # TODO this might not work for CNN
            codes, indices = torch.func.vmap(super().forward)(z)
        else:
            codes, indices = super().forward(z)
        codes = codes.flatten(-2)
        return {"codes": codes, "indices": indices, "z": z, "state": codes}

    def __repr__(self):
        return f"FSQ(levels={self.levels})"


class SimNorm(nn.Module):
    """
    Simplicial normalization.
    Adapted from https://arxiv.org/abs/2204.00616.
    """

    def __init__(self, cfg):
        super().__init__()
        self.dim = cfg.simnorm_dim

    def forward(self, x):
        shp = x.shape
        x = x.view(*shp[:-1], -1, self.dim)
        x = nn.functional.softmax(x, dim=-1)
        return x.view(*shp)

    def __repr__(self):
        return f"SimNorm(dim={self.dim})"


class ContextLinear(nn.Linear):
    """
    Linear layer which can use context as a second input, if desired.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        ctx_dim: Optional[int],
        *args,
        **kwargs,
    ):
        self.use_ctx = ctx_dim is not None
        if self.use_ctx:
            in_features += ctx_dim
        super().__init__(in_features, out_features, *args, **kwargs)

    def forward(self, x: torch.Tensor, ctx: list[torch.Tensor]):
        if self.use_ctx:
            x = torch.cat(ctx + [x], -1)
        return super().forward(x)


class NormedLinear(ContextLinear):
    """
    Linear layer with LayerNorm, Mish activation, and optionally dropout.

    Adapted from https://github.com/tdmpc2/tdmpc2-eval/blob/main/helper.py

    Optionally, conditioning on context can be used. If ctx_dim is not None:
    - The context will be concatenated to the input of the linear layer and
    - The context will be concatenated to the input of the LayerNorm layer.
    """

    def __init__(
        self,
        *args,
        ctx_dim: Optional[int] = None,
        condition_ln: bool = False,
        dropout=0.0,
        act=nn.Mish(inplace=True),
        norm_mode: Optional[str] = "ln",  # "ln" or "bn" or "brn" or None
        norm_after_act: bool = True,
        **kwargs,
    ):
        super().__init__(*args, ctx_dim=ctx_dim, **kwargs)
        if norm_mode == "bn":
            assert ctx_dim is None, "Not implemented"
            self.norm = nn.BatchNorm1d(self.out_features)
        elif norm_mode == "brn":
            assert ctx_dim is None, "Not implemented"
            self.norm = BatchRenorm1d(self.out_features)
        elif norm_mode == "ln":
            ctx_dim_ln = ctx_dim if condition_ln else None
            self.norm = ContextLayerNorm(self.out_features, ctx_dim=ctx_dim_ln)
        elif norm_mode is None:
            assert ctx_dim is None, "Not implemented"
            self.norm = lambda x: x
        else:
            raise NotImplementedError(
                f"norm_mode should be 'ln', 'bn', 'brn' or None, not {norm_mode}"
            )

        self.norm_after_act = norm_after_act
        self.act = act
        self.dropout = nn.Dropout(dropout, inplace=True) if dropout else None
        self.ctx_dim = ctx_dim

    def forward(self, x: torch.Tensor, ctx: list[torch.Tensor]):
        x = super().forward(x, ctx)
        if self.dropout:
            x = self.dropout(x)
        if self.norm_after_act:
            return self.norm(self.act(x), ctx)
        else:
            return self.act(self.norm(x, ctx))

    def __repr__(self):
        repr_dropout = f", dropout={self.dropout.p}" if self.dropout else ""
        return f"NormedLinear(in_features={self.in_features}, \
        out_features={self.out_features}, \
        ctx_dim={self.ctx_dim}, \
        bias={self.bias is not None}{repr_dropout}, \
        act={self.act.__class__.__name__})"


class ContextLayerNorm(nn.LayerNorm):
    """
    LayerNorm layer which can use context as a second input, if desired.
    """

    def __init__(self, normalized_shape: int, ctx_dim: Optional[int], *args, **kwargs):
        self.use_ctx = ctx_dim is not None
        if self.use_ctx:
            normalized_shape += ctx_dim
        super().__init__(normalized_shape=normalized_shape, *args, **kwargs)

    def forward(self, x: torch.Tensor, ctx: list[torch.Tensor]):
        if self.use_ctx:
            x = torch.cat(ctx + [x], -1)
        x = super().forward(x)
        return x


class Ensemble(nn.Module):
    """Vectorized ensemble of modules"""

    def __init__(self, modules, **kwargs):
        super().__init__()

        self.params_dict, self._buffers = stack_module_state(modules)
        self.params = nn.ParameterList([p for p in self.params_dict.values()])

        # Construct a "stateless" version of one of the models. It is "stateless" in
        # the sense that the parameters are meta Tensors and do not have storage.
        base_model = copy.deepcopy(modules[0])
        base_model = base_model.to("meta")

        def fmodel(params, buffers, x1, x2):
            return functional_call(base_model, (params, buffers), (x1, x2))

        # Build in_dims matching the pytree structure
        params_in_dims = {k: 0 for k in self.params_dict.keys()}
        buffers_in_dims = {k: 0 for k in self._buffers.keys()}

        self.vmap = torch.vmap(
            fmodel,
            in_dims=(params_in_dims, buffers_in_dims, None, None),
            randomness="different",
            **kwargs,
        )
        self._repr = str(modules)

    def forward(self, x, ctx):  # Supply modules with context (they may or may not use)
        return self.vmap(self._get_params_dict(), self._buffers, x, ctx)

    def _get_params_dict(self):
        params_dict = {}
        for key, value in zip(self.params_dict.keys(), self.params):
            params_dict.update({key: value})
        return params_dict

    def __repr__(self):
        return "Vectorized " + self._repr


@torch.no_grad()
def orthogonal_init(m):
    """Orthogonal layer initialization."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    # elif isinstance(m, EnsembleLinear):
    #     for w in m.weight.data:
    #         nn.init.orthogonal_(w)
    #     if m.bias is not None:
    #         for b in m.bias.data:
    #             nn.init.zeros_(b)
    elif isinstance(m, (nn.Conv3d, nn.Conv2d, nn.ConvTranspose2d)):
        gain = nn.init.calculate_gain("relu")
        nn.init.orthogonal_(m.weight.data, gain)
        # nn.init.kaiming_uniform_(m.weight.data, mode='fan_in', nonlinearity='relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)


class LinearSchedule:
    def __init__(self, start: float, end: float, num_steps: int):
        self.start = start
        self.end = end
        self.num_steps = num_steps
        self.step_idx = 0
        if num_steps == 0:
            self.values = [end, end]
        else:
            self.values = np.linspace(start, end, num_steps)

    def __call__(self):
        return self.values[self.step_idx]

    def step(self):
        if self.step_idx < self.num_steps - 1:
            self.step_idx += 1


@torch.no_grad()
def calc_rank(name, z):
    """Log rank of latent"""
    rank3 = matrix_rank(z, atol=1e-3, rtol=1e-3)
    rank2 = matrix_rank(z, atol=1e-2, rtol=1e-2)
    rank1 = matrix_rank(z, atol=1e-1, rtol=1e-1)
    condition = cond(z)
    info = {}
    full_rank = z.shape[-1]
    for j, rank in enumerate([rank1, rank2, rank3]):
        rank_percent = rank.item() / full_rank * 100
        info.update({f"{name}-rank-{j}": rank.item()})
        info.update({f"{name}-rank-percent-{j}": rank_percent})
    info.update({f"{name}-cond-num": condition.item()})
    return info


def calc_mean_opt_moments(opt):
    first_moment, second_moment = 0, 0
    for group in opt.param_groups:
        for p in group["params"]:
            state = opt.state[p]
            try:
                first_moment += torch.sum(state["exp_avg"]) / len(state["exp_avg"])
                second_moment += torch.sum(state["exp_avg_sq"]) / len(state["exp_avg"])
            except KeyError:
                pass
    return {"first_moment_mean": first_moment, "second_moment_mean": second_moment}


def seq_to_id(keys: Sequence[str]) -> dict[str, int]:
    """Returns mapping from string identifiers to unique IDs.
    The order of the input sequence determines the ordering of the IDs."""
    str_to_id = {}
    next_id = 0
    for key in keys:
        if key not in str_to_id.keys():
            str_to_id[key] = next_id
            next_id += 1
    return str_to_id


def seq_to_id_naive(keys: Sequence[str]) -> tuple[list, dict[str, int]]:
    """Returns mapping from string identifiers to IDs.
    The twist: Keeps on handing out new ideas, even to already-seen keys
    The order of the input sequence determines the ordering of the IDs."""
    str_to_id = {}
    keys_modified = []
    for next_id, key in enumerate(keys):
        key_naive = f"{key} ({next_id})"  # human-readable + (what the agent sees)
        str_to_id[key_naive] = next_id
        keys_modified.append(key_naive)
    return keys_modified, str_to_id


def symlog(x):
    """
    Symmetric logarithmic function.
    Adapted from https://github.com/danijar/dreamerv3.
    """
    return torch.sign(x) * torch.log(1 + torch.abs(x))


def symexp(x):
    """
    Symmetric exponential function.
    Adapted from https://github.com/danijar/dreamerv3.
    """
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1)


def two_hot(x, cfg):
    """
    Converts a batch of scalars to soft two-hot encoded targets for discrete regression.
    Adapted from https://github.com/nicklashansen/tdmpc2.
    """
    if cfg.num_bins == 0:
        return x
    elif cfg.num_bins == 1:
        return symlog(x)
    x = torch.clamp(symlog(x), cfg.vmin, cfg.vmax).squeeze(1)
    bin_idx = torch.floor((x - cfg.vmin) / cfg.bin_size)
    bin_offset = ((x - cfg.vmin) / cfg.bin_size - bin_idx).unsqueeze(-1)
    soft_two_hot = torch.zeros(x.shape[0], cfg.num_bins, device=x.device, dtype=x.dtype)
    bin_idx = bin_idx.long()
    soft_two_hot = soft_two_hot.scatter(1, bin_idx.unsqueeze(1), 1 - bin_offset)
    soft_two_hot = soft_two_hot.scatter(
        1, (bin_idx.unsqueeze(1) + 1) % cfg.num_bins, bin_offset
    )
    return soft_two_hot


def two_hot_inv(x, cfg):
    """
    Converts a batch of soft two-hot encoded vectors to scalars.
    Adapted from https://github.com/nicklashansen/tdmpc2.
    """
    if cfg.num_bins == 0:
        return x
    elif cfg.num_bins == 1:
        return symexp(x)
    dreg_bins = torch.linspace(
        cfg.vmin, cfg.vmax, cfg.num_bins, device=x.device, dtype=x.dtype
    )
    x = nn.functional.softmax(x, dim=-1)
    x = torch.sum(x * dreg_bins, dim=-1, keepdim=True)
    return symexp(x)


def soft_ce(pred: torch.Tensor, target: torch.Tensor, cfg):
    """
    Computes the cross entropy loss between predictions and soft targets.
    Adapted from https://github.com/nicklashansen/tdmpc2.
    """
    if pred.ndim == 1:
        pred = pred.unsqueeze(-1)
    if target.ndim == 1:
        target = target.unsqueeze(-1)
    pred = nn.functional.log_softmax(pred, dim=-1)
    target = two_hot(target, cfg)
    return -(target * pred).sum(-1)


def rollout_with_ids(
    agent,  # type: iQRL (leads to circular import, thus only implicitly mentioned here)
    env,
    body_id,
    task_id,
    eval_mode: bool,
    max_steps: int,
    return_contiguous: bool = True,
):
    """
    Simplified version of torchrl.envs.common.EnvBase.rollout().

    Implicitly uses break_when_any_done=False i.e. _rollout_nonstop and auto_reset=True.
    """
    tensordict = env.reset()
    tensordicts = []
    tensordict_ = tensordict
    for i in range(max_steps):
        obs = tensordict_["observation"]
        action = agent.select_action(
            obs=obs, body_id=body_id, task_id=task_id, eval_mode=eval_mode
        )
        tensordict_["action"] = action

        if i == max_steps - 1:
            tensordict = env.step(tensordict_)
        else:
            tensordict, tensordict_ = env.step_and_maybe_reset(tensordict_)
        tensordicts.append(tensordict)
        if i == max_steps - 1:
            # we don't truncate as one could potentially continue the run
            break

    if return_contiguous:
        try:
            out_td = torch.stack(tensordicts, len(env.batch_size))
        except RuntimeError as err:
            if (
                "The shapes of the tensors to stack is incompatible" in str(err)
                and env._has_dynamic_specs
            ):
                raise RuntimeError(
                    "The environment specs are dynamic. Call rollout with return_contiguous=False."
                )
            raise
    else:
        out_td = LazyStackedTensorDict.maybe_dense_stack(
            tensordicts, len(env.batch_size)
        )

    return out_td
