import torch
import torch.nn as nn
from typing import Callable, Union


Tensor = torch.Tensor


class _WrapCallable(nn.Module):
    """Wrap a Python function as a torch.nn.Module."""
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x, dt):
        return self.fn(x, dt)


class StrangSplitGating(nn.Module):
    """
    Commutator-safe Strang-splitting MoE gating.

    Applies the stable ABA composition:
        T((1-g)/2 * dt) → V(g * dt) → T((1-g)/2 * dt)
    """

    def __init__(
        self,
        expert_T: Callable[[Tensor, Tensor], Tensor],
        expert_V: Callable[[Tensor, Tensor], Tensor],
    ):
        super().__init__()
        self.expert_T = expert_T if isinstance(expert_T, nn.Module) else _WrapCallable(expert_T)
        self.expert_V = expert_V if isinstance(expert_V, nn.Module) else _WrapCallable(expert_V)

    def forward(
        self,
        rho: Tensor,
        g: Union[float, Tensor],
        dt: Union[float, Tensor],
    ) -> Tensor:

        batch_size = rho.shape[0]
        device, dtype = rho.device, rho.dtype

        # Normalize per-sample scalars
        g = self._prepare_scalar(g, batch_size, device, dtype).clamp(0.0, 1.0)
        dt = self._prepare_scalar(dt, batch_size, device, dtype)

        # Compute effective timesteps
        dt_T = 0.5 * (1.0 - g) * dt     # half-step T
        dt_V = g * dt                  # full-step V

        # Broadcast over state dims
        view_shape = (batch_size,) + (1,) * (rho.ndim - 1)
        dt_T = dt_T.view(view_shape)
        dt_V = dt_V.view(view_shape)

        # Strang-split (ABA): T → V → T
        rho = self.expert_T(rho, dt_T)
        rho = self.expert_V(rho, dt_V)
        rho = self.expert_T(rho, dt_T)

        return rho

    def _prepare_scalar(self, val, batch_size, device, dtype):
        """Broadcast scalar or 1D tensor to (batch,)."""
        val = torch.as_tensor(val, device=device, dtype=dtype)
        if val.ndim == 0:
            return val.expand(batch_size)
        if val.ndim == 1:
            if val.shape[0] == 1:
                return val.expand(batch_size)
            assert val.shape[0] == batch_size, f"Expected batch {batch_size}, got {val.shape}"
            return val
        return val.view(-1)
