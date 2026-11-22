"""手写 SAC(连续动作, squashed-Gaussian, 自动温度), 仅依赖 torch/numpy。"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

LOG_STD_MIN, LOG_STD_MAX = -20.0, 2.0


def _mlp(obs_dim, hidden, out_dim):
    return nn.Sequential(
        nn.Linear(obs_dim, hidden), nn.ReLU(),
        nn.Linear(hidden, hidden), nn.ReLU(),
        nn.Linear(hidden, out_dim),
    )


class SAC:
    def __init__(self, obs_dim, act_dim, hidden=128, lr=3e-4, gamma=0.99, tau=0.005,
                 buffer=100_000, batch=256, warmup=1000, device=None, seed=0):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gamma, self.tau, self.batch, self.warmup = gamma, tau, batch, warmup
        self.act_dim = act_dim
        self.target_entropy = -float(act_dim)
        torch.manual_seed(seed)
        self.actor = _mlp(obs_dim, hidden, 2 * act_dim).to(self.device)
        self.q1 = _mlp(obs_dim + act_dim, hidden, 1).to(self.device)
        self.q2 = _mlp(obs_dim + act_dim, hidden, 1).to(self.device)
        self.q1t = _mlp(obs_dim + act_dim, hidden, 1).to(self.device)
        self.q2t = _mlp(obs_dim + act_dim, hidden, 1).to(self.device)
        self.q1t.load_state_dict(self.q1.state_dict())
        self.q2t.load_state_dict(self.q2.state_dict())
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.opt_a = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.opt_q = torch.optim.Adam(list(self.q1.parameters()) + list(self.q2.parameters()), lr=lr)
        self.opt_al = torch.optim.Adam([self.log_alpha], lr=lr)
        cap = int(buffer)
        self._o = np.zeros((cap, obs_dim), np.float32)
        self._a = np.zeros((cap, act_dim), np.float32)
        self._r = np.zeros(cap, np.float32)
        self._o2 = np.zeros((cap, obs_dim), np.float32)
        self._d = np.zeros(cap, np.float32)
        self._ptr = 0
        self._size = 0
        self._cap = cap

    def _sample_policy(self, o_t):
        out = self.actor(o_t)
        mean, log_std = out.chunk(2, dim=-1)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)
        x = mean + std * torch.randn_like(std)
        a = torch.tanh(x)
        logp = (-0.5 * (((x - mean) / std) ** 2) - log_std - 0.5 * np.log(2 * np.pi))
        logp = logp.sum(-1) - torch.log(1 - a.pow(2) + 1e-6).sum(-1)
        return a, logp

    @torch.no_grad()
    def act(self, obs, deterministic=False):
        o = torch.as_tensor(np.asarray(obs, np.float32)[None], device=self.device)
        if deterministic:
            mean = self.actor(o).chunk(2, dim=-1)[0]
            a = torch.tanh(mean)
        else:
            a, _ = self._sample_policy(o)
        return a.squeeze(0).cpu().numpy()

    def store(self, o, a, r, o2, done):
        i = self._ptr
        self._o[i] = o; self._a[i] = a; self._r[i] = r
        self._o2[i] = o2; self._d[i] = float(done)
        self._ptr = (self._ptr + 1) % self._cap
        self._size = min(self._size + 1, self._cap)

    def _batch(self):
        idx = np.random.randint(0, self._size, size=self.batch)
        d = self.device
        return (torch.as_tensor(self._o[idx], device=d), torch.as_tensor(self._a[idx], device=d),
                torch.as_tensor(self._r[idx], device=d), torch.as_tensor(self._o2[idx], device=d),
                torch.as_tensor(self._d[idx], device=d))

    def update(self):
        if self._size < self.warmup:
            return None
        o, a, r, o2, done = self._batch()
        alpha = self.log_alpha.exp()
        with torch.no_grad():
            a2, logp2 = self._sample_policy(o2)
            q1t = self.q1t(torch.cat([o2, a2], -1)).squeeze(-1)
            q2t = self.q2t(torch.cat([o2, a2], -1)).squeeze(-1)
            q_t = r + self.gamma * (1 - done) * (torch.min(q1t, q2t) - alpha * logp2)
        q1 = self.q1(torch.cat([o, a], -1)).squeeze(-1)
        q2 = self.q2(torch.cat([o, a], -1)).squeeze(-1)
        q_loss = nn.functional.mse_loss(q1, q_t) + nn.functional.mse_loss(q2, q_t)
        self.opt_q.zero_grad(); q_loss.backward()
        nn.utils.clip_grad_norm_(list(self.q1.parameters()) + list(self.q2.parameters()), 1.0)
        self.opt_q.step()

        a1, logp1 = self._sample_policy(o)
        q1_new = self.q1(torch.cat([o, a1], -1)).squeeze(-1)
        q2_new = self.q2(torch.cat([o, a1], -1)).squeeze(-1)
        p_loss = (alpha.detach() * logp1 - torch.min(q1_new, q2_new)).mean()
        self.opt_a.zero_grad(); p_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.opt_a.step()

        a_loss = -(self.log_alpha * (logp1.detach() + self.target_entropy)).mean()
        self.opt_al.zero_grad(); a_loss.backward(); self.opt_al.step()

        with torch.no_grad():
            self._soft(self.q1, self.q1t); self._soft(self.q2, self.q2t)
        return {"q_loss": float(q_loss), "p_loss": float(p_loss),
                "alpha": float(alpha)}

    def _soft(self, src, dst):
        with torch.no_grad():
            for ps, pd in zip(src.parameters(), dst.parameters()):
                pd.mul_(1 - self.tau).add_(ps * self.tau)

    def save(self, path):
        torch.save({"actor": self.actor.state_dict(),
                    "q1": self.q1.state_dict(), "q2": self.q2.state_dict()}, path)

    def load(self, path):
        ck = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(ck["actor"])
        self.q1.load_state_dict(ck["q1"])
        self.q2.load_state_dict(ck["q2"])
