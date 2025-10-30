from functools import partial
from typing import Callable
import torch
from torch import Tensor


def uniform_sample(batch_size: int, device: torch.device, min_t: float = 0.0) -> Tensor:
    return torch.rand(batch_size, device=device) * (1 - 2 * min_t) + min_t


def logit_norm_sample(
    batch_size: int,
    device: torch.device,
    mu: float = -0.4,
    sigma: float = 1.0,
    min_t: float = 0.0,
) -> Tensor:
    normal_samples = mu + sigma * torch.randn(batch_size, device=device)
    return torch.sigmoid(normal_samples) * (1 - 2 * min_t) + min_t


def sample_boundary_mask(
    batch_size: int, device: torch.device, boundary_ratio: float = 0.75
) -> Tensor:
    is_boundary = torch.rand(batch_size, device=device) < boundary_ratio
    return is_boundary


def unordered_interval_sample(
    batch_size: int,
    device: torch.device,
    sample_fn: Callable[[int, torch.device], Tensor],
    boundary_ratio: float = 0.75,
) -> tuple[Tensor, Tensor]:

    is_boundary = sample_boundary_mask(batch_size, device, boundary_ratio)

    t = sample_fn(batch_size, device)
    r = sample_fn(batch_size, device)

    r = torch.where(is_boundary, t, r)

    return t, r


def ordered_interval_sample(
    batch_size: int,
    device: torch.device,
    sample_fn: Callable[[int, torch.device], Tensor],
    boundary_ratio: float = 0.75,
) -> tuple[Tensor, Tensor]:
    is_boundary = sample_boundary_mask(batch_size, device, boundary_ratio)
    t_raw = sample_fn(batch_size, device)
    r_raw = sample_fn(batch_size, device)

    t = torch.minimum(t_raw, r_raw)
    r = torch.maximum(t_raw, r_raw)

    t = torch.where(is_boundary, t_raw, t)
    r = torch.where(is_boundary, t_raw, r)

    return t, r


def length_first_sample(
    batch_size: int,
    device: torch.device,
    boundary_ratio: float = 0.75,
    alpha: float = 0.5,
    beta: float = 0.5,
) -> tuple[Tensor, Tensor]:
    is_boundary = sample_boundary_mask(batch_size, device, boundary_ratio)

    alpha = torch.tensor([alpha], device=device)
    beta = torch.tensor([beta], device=device)

    length = torch.distributions.beta.Beta(alpha, beta).sample((batch_size,)).squeeze()

    t = torch.rand(batch_size, device=device) * (1 - length)
    r = t + length

    t_raw = torch.rand(batch_size, device=device)

    t = torch.where(is_boundary, t_raw, t)
    r = torch.where(is_boundary, t_raw, r)

    return t, r


def ordered_three_point_sample(
    batch_size: int,
    device: torch.device,
    sample_fn: Callable[[int, torch.device], Tensor],
) -> tuple[Tensor, Tensor, Tensor]:
    t_raw = sample_fn(batch_size, device)
    r_raw = sample_fn(batch_size, device)

    t = torch.minimum(t_raw, r_raw)
    r = torch.maximum(t_raw, r_raw)

    w = torch.rand(batch_size, device=device)
    s = w * r + (1 - w) * t

    return t, s, r


def length_first_three_point_sample(
    batch_size: int,
    device: torch.device,
    alpha: float = 0.5,
    beta: float = 0.5,
) -> tuple[Tensor, Tensor, Tensor]:

    t, r = length_first_sample(batch_size, device, 0.0, alpha, beta)

    w = torch.rand(batch_size, device=device)
    s = w * r + (1 - w) * t

    return t, s, r


def create_time_sampler(cfg) -> Callable[[int, torch.device], Tensor]:
    if cfg.type == "uniform":
        sample_fn = partial(uniform_sample, min_t=cfg.min_t)

    elif cfg.type == "lognorm":
        sample_fn = partial(
            logit_norm_sample, mu=cfg.mu, sigma=cfg.sigma, min_t=cfg.min_t
        )

    # If we need only single time point sample, return the current time sampler
    need_interval_sample = "interval_sample" in cfg
    need_three_sample = "three_sample" in cfg
    if not need_interval_sample and not need_three_sample:
        return sample_fn

    # If we need interval sample or three point sample, build it with the current time sampler.
    if need_interval_sample and cfg.interval_sample == "ordered":
        sample_fn = partial(
            ordered_interval_sample,
            sample_fn=sample_fn,
            boundary_ratio=cfg.boundary_ratio,
        )
    elif need_interval_sample and cfg.interval_sample == "unordered":
        sample_fn = partial(
            unordered_interval_sample,
            sample_fn=sample_fn,
            boundary_ratio=cfg.boundary_ratio,
        )
    elif need_interval_sample and cfg.interval_sample == "interval_length_first":
        sample_fn = partial(
            length_first_sample,
            boundary_ratio=cfg.boundary_ratio,
            alpha=cfg.alpha,
            beta=cfg.beta,
        )
    elif need_three_sample and cfg.three_sample == "interval_length_first":
        sample_fn = partial(
            length_first_three_point_sample,
            alpha=cfg.alpha,
            beta=cfg.beta,
        )

    elif need_three_sample and cfg.three_sample == "ordered":
        sample_fn = partial(
            ordered_three_point_sample,
            sample_fn=sample_fn,
        )
    else:
        raise ValueError(f"Invalid time sampler type: {cfg.three_sample}")

    return sample_fn
