# mixture_igso3_dataset.py
# Mixture of IGSO(3) with two modes — PyTorch Dataset
# - Analytic ground-truth velocity for geodesic flow-matching
# - Mixture sampler
# - MMD (RBF on geodesic distance) utilities

from __future__ import annotations
import math
import torch
from torch.utils.data import Dataset

Tensor = torch.Tensor


# ---------- SO(3) utilities ----------
def hat(v: Tensor) -> Tensor:
    """Skew (hat): (...,3) -> (...,3,3)"""
    x, y, z = v.unbind(-1)
    Z = torch.zeros(v.shape[:-1] + (3, 3), dtype=v.dtype, device=v.device)
    Z[..., 0, 1], Z[..., 0, 2] = -z, y
    Z[..., 1, 0], Z[..., 1, 2] = z, -x
    Z[..., 2, 0], Z[..., 2, 1] = -y, x
    return Z


def _safe_norm(x: Tensor, eps: float = 1e-12) -> Tensor:
    return torch.linalg.norm(x, dim=-1, keepdim=True).clamp_min(eps)


def exp_so3(omega: Tensor) -> Tensor:
    """Rodrigues exponential. omega: (...,3) axis-angle vector -> (...,3,3)"""
    theta = _safe_norm(omega)  # (...,1)
    u = omega / theta  # (...,3)
    K = hat(u)  # (...,3,3)
    th = theta[..., 0].unsqueeze(-1).unsqueeze(-1)  # (...,1,1)
    I = torch.eye(3, dtype=omega.dtype, device=omega.device)
    return I + torch.sin(th) * K + (1 - torch.cos(th)) * (K @ K)


def log_so3(R: Tensor, eps: float = 1e-6) -> Tensor:
    """Matrix logarithm -> axis-angle vector (...,3)
    Stable except very near pi (clip trace & sin(theta) denom)."""
    # clip trace for numerical stability
    tr = (R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]).clamp(-1 + eps, 3 - eps)
    theta = torch.acos((tr - 1.0) / 2.0)  # (...,)
    # axis from antisymmetric part
    v = torch.stack(
        [
            R[..., 2, 1] - R[..., 1, 2],
            R[..., 0, 2] - R[..., 2, 0],
            R[..., 1, 0] - R[..., 0, 1],
        ],
        dim=-1,
    )  # (...,3)
    denom = (2.0 * torch.sin(theta).clamp_min(eps)).unsqueeze(-1)  # (...,1)
    u = v / denom  # (...,3)
    return (theta.unsqueeze(-1)) * u  # (...,3)


def geodesic_dist(R: Tensor, S: Tensor, eps: float = 1e-6) -> Tensor:
    """d_SO(3)(R,S) = ||log(R^T S)||_2. Returns (...,)"""
    xi = log_so3(R.transpose(-1, -2) @ S, eps=eps)
    return torch.linalg.norm(xi, dim=-1)


def sample_haar_R(
    B: int, device: str | torch.device = "cpu", dtype=torch.float32
) -> Tensor:
    """Quick Haar-ish sampler via QR with det=+1."""
    A = torch.randn(B, 3, 3, device=device, dtype=dtype)
    Q, _ = torch.linalg.qr(A)
    s = torch.sign(torch.linalg.det(Q)).view(B, 1, 1)
    return Q * s


# ---------- Schedules ----------
class Schedule:
    """kappa(t) and derivative."""

    def __init__(self, kind: str = "linear", alpha: float = 5.0):
        self.kind = kind
        self.alpha = alpha

    def kappa(self, t: Tensor) -> Tensor:
        if self.kind == "linear":
            return t
        elif self.kind == "quadratic":
            return t * t
        elif self.kind == "exp":  # 1 - exp(-alpha t)
            return 1.0 - torch.exp(-self.alpha * t)
        else:
            raise ValueError(f"Unknown schedule: {self.kind}")

    def kappa_prime(self, t: Tensor) -> Tensor:
        if self.kind == "linear":
            return torch.ones_like(t)
        elif self.kind == "quadratic":
            return 2.0 * t
        elif self.kind == "exp":
            return self.alpha * torch.exp(-self.alpha * t)
        else:
            raise ValueError(f"Unknown schedule: {self.kind}")


# ---------- Mixture-of-IGSO(3) Dataset ----------
class MixtureIGSO3Dataset(Dataset):
    """
    Two-mode mixture of IGSO(3) for flow-matching on SO(3).
    Each item:
      - sample mode k in {0,1} with weights w
      - sample R1 ~ IGSO3(mu_k, sigma_k^2) via small-angle Gaussian in algebra
      - sample R0 (Haar or I)
      - sample t in (0,1)
      - Rt = R0 exp(kappa(t) * xi), xi = log(R0^T R1)
      - GT left-trivialized velocity Omega_vec = kappa'(t) * xi (axis-angle in algebra)
      - GT Eulerian velocity matrix v_t = Rt @ hat(Omega_vec)
    """

    def __init__(
        self,
        N: int = 10000,
        mu1: Tensor | None = None,
        mu2: Tensor | None = None,
        sigma1: float = 0.3,
        sigma2: float = 0.3,
        w1: float = 0.5,
        haar_R0: bool = True,
        schedule: Schedule | None = None,
        device: str | torch.device = "cpu",
        dtype=torch.float32,
        theta_clip: float = 1e-3,
        seed: int | None = None,
    ):
        """
        Args:
          N: virtual dataset length (on-the-fly sampling)
          mu1, mu2: (3,3) rotation matrices (defaults: I, RotZ(120°))
          sigma1, sigma2: scalar std for small-angle Gaussian (axis-angle)
          w1: mixture weight for mode 1 (mode 2 has 1-w1)
          haar_R0: if False, R0=I (useful for debugging)
          schedule: time schedule (default linear)
          theta_clip: avoid theta ~ pi by clipping to (pi - theta_clip)
        """
        super().__init__()
        self.N = N
        self.device = torch.device(device)
        self.dtype = dtype
        self.haar_R0 = haar_R0
        self.theta_max = math.pi - theta_clip
        self.sigmas = torch.tensor(
            [sigma1, sigma2], device=self.device, dtype=self.dtype
        )
        self.weights = torch.tensor(
            [w1, 1.0 - w1], device=self.device, dtype=self.dtype
        )
        self.cat = torch.distributions.Categorical(self.weights)
        if schedule is None:
            schedule = Schedule("linear")
        self.sched = schedule

        if seed is not None:
            g = torch.Generator(device=self.device)
            g.manual_seed(seed)
            self._generator = g
        else:
            self._generator = None

        I = torch.eye(3, dtype=self.dtype, device=self.device)
        if mu1 is None:
            mu1 = I.clone()
        if mu2 is None:
            # default: 120 degrees around z
            angle = 2.0 * math.pi / 3.0
            mu2 = exp_so3(
                torch.tensor([0.0, 0.0, angle], device=self.device, dtype=self.dtype)
            )
        self.mus = torch.stack(
            [mu1.to(self.device, self.dtype), mu2.to(self.device, self.dtype)], dim=0
        )  # (2,3,3)

    def __len__(self) -> int:
        return self.N

    @torch.no_grad()
    def _randn(self, *shape) -> Tensor:
        if self._generator is None:
            return torch.randn(*shape, device=self.device, dtype=self.dtype)
        return torch.randn(
            *shape, device=self.device, dtype=self.dtype, generator=self._generator
        )

    @torch.no_grad()
    def _rand(self, *shape) -> Tensor:
        if self._generator is None:
            return torch.rand(*shape, device=self.device, dtype=self.dtype)
        return torch.rand(
            *shape, device=self.device, dtype=self.dtype, generator=self._generator
        )

    @torch.no_grad()
    def _sample_mode(self, B: int) -> Tensor:
        # returns (B,) indices in {0,1}
        if self._generator is None:
            return self.cat.sample((B,))
        # Manual categorical sampling using CDF and uniform noise with the provided generator
        u = torch.rand(
            B, device=self.device, dtype=self.dtype, generator=self._generator
        )
        cdf = torch.cumsum(self.weights, dim=0)  # (2,)
        idx = torch.searchsorted(cdf, u)  # (B,)
        return idx

    @torch.no_grad()
    def _sample_R1_from_mode(self, k: Tensor) -> Tensor:
        """Small-angle IGSO(3) approx: R1 = mu_k exp(xi), xi ~ N(0, sigma_k^2 I), clipped at pi - eps."""
        B = k.shape[0]
        sigma = self.sigmas[k].unsqueeze(-1)  # (B,1)
        xi = self._randn(B, 3) * sigma  # (B,3)
        th = _safe_norm(xi)  # (B,1)
        # clip angle to avoid ~pi
        th_clipped = torch.minimum(
            th,
            torch.tensor(
                self.theta_max, device=self.device, dtype=self.dtype
            ).expand_as(th),
        )
        xi = xi * (th_clipped / th)
        R1 = self.mus[k] @ exp_so3(xi)  # (B,3,3)
        return R1, xi

    @torch.no_grad()
    def _sample_R0(self, B: int) -> Tensor:
        if self.haar_R0:
            return sample_haar_R(B, device=self.device, dtype=self.dtype)
        I = torch.eye(3, dtype=self.dtype, device=self.device)
        return I.expand(B, 3, 3).clone()

    @torch.no_grad()
    def __getitem__(self, idx: int):
        # batchless sampling for a single item
        k = self._sample_mode(1)  # (1,)
        R1, _ = self._sample_R1_from_mode(k)  # (1,3,3)
        R0 = self._sample_R0(1)  # (1,3,3)
        t = self._rand(1)  # (1,)

        # geodesic pair
        xi = log_so3(R0.transpose(-1, -2) @ R1)  # (1,3)
        Rt = R0 @ exp_so3(self.sched.kappa(t).unsqueeze(-1) * xi)  # (1,3,3)

        # GT velocity
        Omega_vec = self.sched.kappa_prime(t).unsqueeze(-1) * xi  # (1,3)
        v_t = Rt @ hat(Omega_vec)  # (1,3,3)

        sample = {
            "Rt": Rt.squeeze(0),  # (3,3)
            "t": t.squeeze(0),  # ()
            "R1": R1.squeeze(0),  # (3,3)
            "R0": R0.squeeze(0),  # (3,3)
            "mode": k.item(),  # int
            "Omega_vec": Omega_vec.squeeze(0),  # (3,)  left-trivialized GT
            "v_t": v_t.squeeze(0),  # (3,3) Eulerian GT
        }
        return sample

    # ---------- Public utilities ----------

    @torch.no_grad()
    def sample_mixture(self, B: int) -> tuple[Tensor, Tensor]:
        """Sample R ~ sum_k w_k IGSO3(mu_k, sigma_k^2) (small-angle approx).
        Returns (R, k) with shapes (B,3,3), (B,)"""
        k = self._sample_mode(B)
        R, _ = self._sample_R1_from_mode(k)  # identical to R1 sampling
        return R, k


# ---------- MMD utilities on SO(3) ----------
@torch.no_grad()
def rbf_kernel_R(R: Tensor, S: Tensor, sigma: float = 0.5) -> Tensor:
    """
    RBF kernel on SO(3) using geodesic distance:
      k(R,S) = exp( - d(R,S)^2 / (2 sigma^2) )
    Shapes:
      R: (N,3,3), S: (M,3,3) -> (N,M)
    """
    # pairwise dists: expand & compute log(R^T S)
    N, M = R.shape[0], S.shape[0]
    Rt = R.transpose(-1, -2).unsqueeze(1).expand(N, M, 3, 3)  # (N,M,3,3)
    Srep = S.unsqueeze(0).expand(N, M, 3, 3)  # (N,M,3,3)
    xi = log_so3(Rt @ Srep)  # (N,M,3)
    d2 = (xi**2).sum(dim=-1)  # (N,M)
    return torch.exp(-d2 / (2.0 * (sigma**2)))


@torch.no_grad()
def mmd2_unbiased_R(X: Tensor, Y: Tensor, sigma: float = 0.5) -> Tensor:
    """
    Unbiased MMD^2 with RBF kernel on SO(3).
    X: (N,3,3), Y: (M,3,3)
    """
    N, M = X.shape[0], Y.shape[0]
    Kxx = rbf_kernel_R(X, X, sigma)
    Kyy = rbf_kernel_R(Y, Y, sigma)
    Kxy = rbf_kernel_R(X, Y, sigma)

    # zero diagonal terms for unbiased estimate
    Kxx = Kxx - torch.diag_embed(torch.diagonal(Kxx, dim1=-2, dim2=-1))
    Kyy = Kyy - torch.diag_embed(torch.diagonal(Kyy, dim1=-2, dim2=-1))

    mmd2 = (
        Kxx.sum() / (N * (N - 1) + 1e-12)
        + Kyy.sum() / (M * (M - 1) + 1e-12)
        - 2.0 * Kxy.mean()
    )
    return mmd2


# ---------- Example defaults ----------
def default_two_modes(device="cpu", dtype=torch.float32):
    """Convenience getter for (mu1, mu2) used above."""
    I = torch.eye(3, device=device, dtype=dtype)
    angle = 2.0 * math.pi / 3.0
    mu2 = exp_so3(torch.tensor([0.0, 0.0, angle], device=device, dtype=dtype))
    return I, mu2


from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader


class MIGDataModule(LightningDataModule):
    def __init__(
        self,
        train_dataset: MixtureIGSO3Dataset,
        val_dataset: MixtureIGSO3Dataset,
        batch_size: int = 128,
    ):
        super().__init__()
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.batch_size = batch_size

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)

    def predict_dataloader(self):
        raise NotImplementedError(
            "Predict dataloader not implemented for MIGDataModule"
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Compute MMD^2 of MixtureIGSO3Dataset: mixture vs mixture and mixture vs uniform."
    )
    parser.add_argument(
        "--num_samples", type=int, default=512, help="Number of samples per set"
    )
    parser.add_argument(
        "--sigma", type=float, default=0.5, help="RBF kernel sigma on SO(3)"
    )
    parser.add_argument(
        "--device", type=str, default="cpu", help="Torch device, e.g., 'cpu' or 'cuda'"
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    args = parser.parse_args()

    # Set global seed as well for reproducibility of Haar sampling
    torch.manual_seed(args.seed)

    dataset = MixtureIGSO3Dataset(device=args.device, seed=args.seed)
    with torch.no_grad():
        X, _ = dataset.sample_mixture(args.num_samples)
        Y, _ = dataset.sample_mixture(args.num_samples)
        mmd2_mm = mmd2_unbiased_R(X, Y, sigma=args.sigma)
        U = sample_haar_R(args.num_samples, device=args.device, dtype=dataset.dtype)
        mmd2_mu = mmd2_unbiased_R(X, U, sigma=args.sigma)
    print(
        f"MMD^2 (mixture vs mixture): {mmd2_mm.item():.6e} | N={args.num_samples} sigma={args.sigma}"
    )
    print(
        f"MMD^2 (mixture vs uniform): {mmd2_mu.item():.6e} | N={args.num_samples} sigma={args.sigma}"
    )
