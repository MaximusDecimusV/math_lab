"""Training loops. Seeds: numpy via mathlab_shared.rng_from_seed, torch via manual_seed."""

from __future__ import annotations

import time

import numpy as np
import torch

from chaos_dyn.data import Dataset
from chaos_dyn.models import DirectMLP, StepMLP
from mathlab_shared import rng_from_seed


def _t(x: np.ndarray) -> torch.Tensor:
    return torch.as_tensor(x, dtype=torch.float32)


def _schedule(opt, total: int):
    return torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=opt.param_groups[0]["lr"], total_steps=total
    )


def train_step_model(
    model: StepMLP,
    train: Dataset,
    iters: int = 20_000,
    batch: int = 2048,
    lr: float = 2e-3,
    unroll: int = 1,
    seed: int = 0,
    log_every: int = 2000,
) -> list[float]:
    """Fit s_{k+1} from s_k. With unroll = K > 1 the model is rolled out K steps inside the
    loss (gradients flow through the rollout), which penalises error accumulation."""
    torch.manual_seed(seed)
    rng = rng_from_seed(seed)
    trajs, cond = _t(train.trajs), _t(train.cond)
    M, T = trajs.shape[:2]
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = _schedule(opt, iters)
    losses, t0 = [], time.time()
    for it in range(iters):
        i = torch.as_tensor(rng.integers(0, M, batch))
        k = torch.as_tensor(rng.integers(0, T - unroll, batch))
        s = trajs[i, k]
        c = cond[i]
        loss = 0.0
        for u in range(1, unroll + 1):
            s = model(s, c)
            loss = loss + (((s - trajs[i, k + u]) / model.d_sd) ** 2).mean()
        loss = loss / unroll
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        losses.append(float(loss))
        if log_every and (it + 1) % log_every == 0:
            print(
                f"  step-model it {it + 1}: loss {np.mean(losses[-log_every:]):.3e} ({time.time() - t0:.0f}s)"
            )
    return losses


def train_direct_model(
    model: DirectMLP,
    train: Dataset,
    iters: int = 20_000,
    batch: int = 256,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    seed: int = 0,
    log_every: int = 2000,
) -> list[float]:
    """Fit the whole N-step trajectory from s_0 in one shot (normalised displacement MSE)."""
    torch.manual_seed(seed)
    rng = rng_from_seed(seed)
    trajs, cond = _t(train.trajs), _t(train.cond)
    target = (trajs[:, 1:] - trajs[:, :1] - model.o_mu) / model.o_sd
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = _schedule(opt, iters)
    losses, t0 = [], time.time()
    for it in range(iters):
        i = torch.as_tensor(rng.integers(0, len(trajs), batch))
        loss = ((model.disp_normalised(trajs[i, 0], cond[i]) - target[i]) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        losses.append(float(loss))
        if log_every and (it + 1) % log_every == 0:
            print(
                f"  direct-model it {it + 1}: loss {np.mean(losses[-log_every:]):.3e} ({time.time() - t0:.0f}s)"
            )
    return losses


@torch.no_grad()
def predict(model, data: Dataset, n_steps: int | None = None) -> np.ndarray:
    model.eval()
    n = n_steps or data.trajs.shape[1] - 1
    out = model.rollout(_t(data.trajs[:, 0]), _t(data.cond), n).double().numpy()
    model.train()
    return out


def load_or_train(model: torch.nn.Module, path, train_fn) -> torch.nn.Module:
    """Reuse a saved state_dict (outputs/ is gitignored) or train and save one."""
    from pathlib import Path

    path = Path(path)
    if path.exists():
        model.load_state_dict(torch.load(path))
        print(f"loaded {path.name}")
    else:
        train_fn(model)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), path)
    return model
