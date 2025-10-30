import torch
from dataclasses import dataclass, field
from data import so3_utils


class ExpSchedule:
    def __init__(self, exp_rate):
        self.exp_rate = exp_rate

    def __call__(self, t):
        return 1 - torch.exp(-t * self.exp_rate)


class LinearSchedule:
    def __call__(self, t):
        return t


class SO3Interpolant:
    def __init__(self, cfg):
        self.cfg = cfg

        if self.cfg.sample_schedule == "exp":
            self.kappa = ExpSchedule(self.cfg.exp_rate)
        elif self.cfg.sample_schedule == "linear":
            self.kappa = LinearSchedule()
        else:
            raise ValueError(f"Invalid sample schedule: {self.cfg.sample_schedule}")

    def sample_xt(
        self, x_0: torch.Tensor, x_1: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        kappa_t = self.kappa(t)
        x_t = so3_utils.geodesic_t(kappa_t[..., None], x_1, x_0)
        return x_t

    def cond_vf(
        self, t: torch.Tensor, x_t: torch.Tensor, x_1: torch.Tensor
    ) -> torch.Tensor:
        v = so3_utils.calc_rot_vf(x_t, x_1)

        if self.cfg.sample_schedule == "exp":
            return self.cfg.exp_rate * v
        elif self.cfg.sample_schedule == "linear":
            return v / ((1 - t)[..., None] + 1e-6)
        else:
            raise ValueError(f"Invalid schedule: {self.cfg.sample_schedule}")

    def euler_step_with_x1_prediction(
        self,
        d_t: torch.Tensor,
        t: torch.Tensor,
        x_1_pred: torch.Tensor,
        x_t: torch.Tensor,
    ) -> torch.Tensor:
        if self.cfg.sample_schedule == "linear":
            scaling = 1 / (1 - t)
        elif self.cfg.sample_schedule == "exp":
            scaling = self.cfg.exp_rate
        else:
            raise ValueError(f"Unknown sample schedule {self.cfg.sample_schedule}")
        return so3_utils.geodesic_t(scaling * d_t, x_1_pred, x_t)

    def euler_step(
        self,
        d_t: torch.Tensor,
        x_t: torch.Tensor,
        vf: torch.Tensor,
    ) -> torch.Tensor:
        return so3_utils.geodesic_t(d_t, None, x_t, vf)
