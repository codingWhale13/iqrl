#!/usr/bin/env python3
import copy
from typing import List, Optional, Sequence, Union

import numpy as np
import torch
import torch.nn as nn
from torch.func import functional_call, stack_module_state
from torch.linalg import cond, matrix_rank
from vector_quantize_pytorch import FSQ as _FSQ


def soft_update_params(model, model_target, tau: float):
    """Update slow-moving average of online network (target network) at rate tau."""
    with torch.no_grad():
        for params, params_target in zip(model.parameters(), model_target.parameters()):
            params_target.data.lerp_(params.data, tau)
            # One below is from CleanRL
            # params_target.data.copy_(tau * params.data + (1 - tau) * params_target.data)


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


class ContextLinear(nn.Linear):
    """
    Linear layer which can use context as a second input, if desired.
    """

    def __init__(self, in_dim: int, out_dim: int, ctx_dim: int = 0, *args, **kwargs):
        self.use_ctx = ctx_dim > 0
        if self.use_ctx:
            in_dim += ctx_dim
        super().__init__(in_dim, out_dim, *args, **kwargs)

    def forward(self, x: torch.Tensor, ctx: list[torch.Tensor]) -> torch.Tensor:
        if self.use_ctx:
            x = torch.cat(ctx + [x], -1)
        return super().forward(x)


class ContextLayerNorm(nn.LayerNorm):
    """
    LayerNorm which can use context as a second input, if desired.
    """

    def __init__(self, normalized_shape: int, ctx_dim: int = 0, *args, **kwargs):
        self.use_ctx = ctx_dim > 0
        if self.use_ctx:
            normalized_shape += ctx_dim
        super().__init__(normalized_shape=normalized_shape, *args, **kwargs)

    def forward(self, x: torch.Tensor, ctx: list[torch.Tensor]) -> torch.Tensor:
        if self.use_ctx:
            x = torch.cat(ctx + [x], -1)
        return super().forward(x)


class AdaptiveLayerNormOriginal(nn.Module):
    """
    Adaptive LayerNorm a.k.a. AdaNorm, as introduced by Xu (2019).
    Adapted from https://github.com/lancopku/AdaNorm/tree/master/machine%20translation/fairseq/modules/layer_norm.py
    """

    def __init__(self, eps=1e-5, adanorm_scale=1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.adanorm_scale = adanorm_scale
        self.eps = eps

    def forward(self, x: torch.Tensor, _: list[torch.Tensor]) -> torch.Tensor:
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        x = x - mean
        mean = x.mean(-1, keepdim=True)
        graNorm = (1 / 10 * (x - mean) / (std + self.eps)).detach()
        x_norm = (x - x * graNorm) / (std + self.eps)

        return x_norm * self.adanorm_scale


class AdaptiveLayerNorm(nn.Module):
    """
    Conditional / Adaptive LayerNorm.
    y = (LN(x, affine=False)) * gamma(ctx) + beta(ctx).
    Adapted from https://github.com/eloialonso/diamond/blob/main/src/models/blocks.py

    We replace bias and gain of LayerNorm by linear layers, mapping ctx_dim -> layer_dim
    NOTE: This is closer to the implementation of LayerNorm than the one of AdaNorm.
    """

    def __init__(self, in_dim: int, ctx_dim: int, eps=1e-5, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.ln = nn.LayerNorm(in_dim, eps=eps, elementwise_affine=False)  # No params
        self.linear = nn.Linear(ctx_dim, in_dim * 2)

    def forward(self, x: torch.Tensor, ctx: list[torch.Tensor]) -> torch.Tensor:
        x_norm = self.ln(x)
        scale, shift = self.linear(torch.cat(ctx, -1)).chunk(2, dim=-1)

        return x_norm * (1 + scale) + shift


class FiLM(nn.Module):
    """
    Learning scale and bias based on context. No normalization.
    """

    def __init__(self, in_dim: int, ctx_dim: int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.linear = nn.Linear(ctx_dim, in_dim * 2)

    def forward(self, x: torch.Tensor, ctx: list[torch.Tensor]) -> torch.Tensor:
        scale, shift = self.linear(torch.cat(ctx, -1)).chunk(2, dim=-1)
        return x * (1 + scale) + shift


class NormedContextLinear(ContextLinear):
    """
    Linear layer, allowing:
    - Context to be concatenated to input tensor.
    - Normalization of the activations.
    Uses Mish activation by default, and optionally dropout.

    Adapted from https://github.com/tdmpc2/tdmpc2-eval/blob/main/helper.py

    If ctx_dim > 0, we concatenate the context to the input of the linear layer.
    If norm_mode is cln, aln, or FiLM, the normalization also gets context-conditioned.
    """

    def __init__(
        self,
        *args,
        ctx_dim: int = 0,
        dropout=0.0,
        norm_mode: Optional[str] = None,  # None, "ln", "cln", "aln", or "FiLM"
        act_fn=nn.Mish(inplace=True),
        norm_after_act: bool = True,
        **kwargs,
    ):
        super().__init__(*args, ctx_dim=ctx_dim, **kwargs)
        self.ctx_dim = ctx_dim
        self.dropout = nn.Dropout(dropout, inplace=True) if dropout else None
        self.norm_mode = norm_mode
        if norm_mode == "ln":  # LayerNorm, no context-conditioning
            self.norm = ContextLayerNorm(self.out_features, ctx_dim=0)
        elif norm_mode == "aln":  # Adaptive LayerNorm, no context-conditioning
            self.norm = AdaptiveLayerNorm(self.out_features, ctx_dim)
        elif norm_mode == "cln":  # LayerNorm, conditioned on context by concatenation
            self.norm = ContextLayerNorm(self.out_features, ctx_dim=ctx_dim)
        elif norm_mode == "FiLM":  # FiLM
            self.norm = FiLM(self.out_features, ctx_dim=ctx_dim)
        elif norm_mode is None:
            self.norm = lambda x, _: x  # Ignore context (second argument), don't use LN
        else:
            raise NotImplementedError(
                f"norm_mode can be None, 'ln', 'cln', 'aln', or 'FiLM', not {norm_mode}"
            )
        self.act_fn = act_fn
        self.norm_after_act = norm_after_act

    def forward(self, x: torch.Tensor, ctx: list[torch.Tensor]) -> torch.Tensor:
        x = super().forward(x, ctx)
        if self.dropout:
            x = self.dropout(x)
        if self.norm_after_act:
            return self.norm(self.act_fn(x), ctx)
        else:
            return self.act_fn(self.norm(x, ctx))

    def __repr__(self):
        repr_dropout = f", dropout={self.dropout.p}" if self.dropout else ""
        if self.norm_mode is None:
            norm_mode = "Identity"
        elif self.norm_mode == "ln":
            norm_mode = "LayerNorm"
        elif self.norm_mode == "cln":
            norm_mode = "ConcatLayerNorm"
        elif self.norm_mode == "aln":
            norm_mode = "AdaptiveLayerNorm"
        elif self.norm_mode == "FiLM":
            norm_mode = "FiLM"
        else:
            norm_mode = "UnknownNorm"
        return f"{norm_mode}(in_features={self.in_features}, \
        out_features={self.out_features}, \
        ctx_dim={self.ctx_dim}, \
        bias={self.bias is not None}{repr_dropout}, \
        act={self.act_fn.__class__.__name__})"


class ContextSequential(nn.Sequential):
    def forward(self, x: torch.Tensor, ctx: list[torch.Tensor]) -> torch.Tensor:
        for module in self:
            x = module(x, ctx)
        return x


def mlp(
    in_dim: int,
    mlp_dims: Union[int, list[int]],
    out_dim: int,
    ctx_dim: int = 0,
    dropout=0.0,
    condition_layer: Optional[str] = None,  # None, "first", or "all"
    norm_mode: Optional[str] = None,  # None, "ln", "cln", "aln", or "FiLM"
    act_fn=None,
    norm_after_act: bool = False,
):
    """
    MLP with Mish activations and optionally:
    - Use dropout in first layer.
    - Use normalization in hidden layer.

    Adapted from https://github.com/tdmpc2/tdmpc2-eval/blob/main/helper.py

    If ctx_dim>0, the specified layers get context-conditioned by concatenation:
    - The in_dim increases by ctx_dim, if condition in ["first", "all"].
    - All mlp_dims increase by ctx_dim, if condition == "all".
    """
    if isinstance(mlp_dims, int):
        mlp_dims = [mlp_dims]

    use_cln = norm_mode == "cln"
    dims = [int(in_dim)] + mlp_dims + [int(out_dim)]
    mlp = nn.ModuleList()

    # Add input layer
    mlp.append(
        NormedContextLinear(
            dims[0],
            dims[1],
            ctx_dim=ctx_dim if condition_layer in ["first", "all"] else 0,
            dropout=dropout,
            norm_mode=norm_mode,
            norm_after_act=norm_after_act,
        )
    )
    add_to_in_dim = 0  # Conditioning LayerNorm changes outgoing dims of layers
    if condition_layer in ["first", "all"] and use_cln:
        add_to_in_dim = ctx_dim

    # Add hidden layer(s)
    for i in range(1, len(dims) - 2):
        mlp.append(
            NormedContextLinear(
                dims[i] + add_to_in_dim,
                dims[i + 1],
                ctx_dim=ctx_dim if condition_layer == "all" else 0,
                norm_mode=norm_mode,
                norm_after_act=norm_after_act,
            )
        )
        add_to_in_dim = ctx_dim if (condition_layer == "all" and use_cln) else 0

    # Add output layer
    mlp.append(
        NormedContextLinear(
            dims[-2] + add_to_in_dim,
            dims[-1],
            ctx_dim=ctx_dim if condition_layer == "all" else 0,
            norm_mode=norm_mode,
            act_fn=act_fn,
            norm_after_act=norm_after_act,
        )
        if act_fn is not None
        else ContextLinear(
            in_dim=dims[-2] + add_to_in_dim,
            out_dim=dims[-1],
            ctx_dim=ctx_dim if condition_layer == "all" else 0,
        )
    )

    return ContextSequential(*mlp)


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
