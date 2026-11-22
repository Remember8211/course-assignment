"""手写 PPO(连续动作), 仅依赖 torch/numpy。"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


def _mlp(obs_dim, hidden, out_dim):
    return nn.Sequential(
        nn.Linear(obs_dim, hidden), nn.ReLU(),
        nn.Linear(hidden, hidden), nn.ReLU(),
        nn.Linear(hidden, out_dim),
    )


class PPO:
    def __init__(self, obs_dim, act_dim, hidden=128, lr=3e-4, gamma=0.99, lam=0.95,
                 clip=0.2, epochs=10, minibatch=64, rollout=2048, ent_coef=0.0,
                 vf_coef=0.5, device=None, seed=0):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gamma, self.lam, self.clip = gamma, lam, clip
        self.epochs, self.minibatch, self.rollout = epochs, minibatch, rollout
        self.ent_coef, self.vf_coef = ent_coef, vf_coef
        self.act_dim = act_dim
        torch.manual_seed(seed)
        self.actor = _mlp(obs_dim, hidden, act_dim).to(self.device)
        self.log_std = nn.Parameter(torch.full((act_dim,), -0.5, device=self.device))
        self.critic = _mlp(obs_dim, hidden, 1).to(self.device)
        self.opt_a = torch.optim.Adam(list(self.actor.parameters()) + [self.log_std], lr=lr)
        self.opt_c = torch.optim.Adam(self.critic.parameters(), lr=lr)
        self._buf = {k: [] for k in ("o", "a", "logp", "r", "done", "v")}

    def _dist(self, obs_t):
        mean = self.actor(obs_t)
        std = torch.exp(torch.clamp(self.log_std, -3.0, -0.5)).expand_as(mean)
        return mean, std

    def _sample_squashed(self, mean, std):
        x = mean + std * torch.randn_like(std)
        a = torch.tanh(x)
        logp = self._logp(mean, std, x) - torch.log(1 - a.pow(2) + 1e-6).sum(-1)
        return a, logp

    def _logp(self, mean, std, a):
        return (-0.5 * (((a - mean) / std) ** 2) - torch.log(std)
                - 0.5 * np.log(2 * np.pi)).sum(dim=-1)

    @torch.no_grad()
    def act_train(self, obs):
        o = torch.as_tensor(np.asarray(obs, np.float32)[None], device=self.device)
        mean, std = self._dist(o)
        a, logp = self._sample_squashed(mean, std)
        val = self.critic(o)
        return (a.squeeze(0).cpu().numpy(), float(logp), float(val.squeeze()))

    @torch.no_grad()
    def act(self, obs, deterministic=False):
        if deterministic:
            o = torch.as_tensor(np.asarray(obs, np.float32)[None], device=self.device)
            mean, _ = self._dist(o)
            return torch.tanh(mean).squeeze(0).cpu().numpy()
        a, _, _ = self.act_train(obs)
        return a

    @torch.no_grad()
    def value(self, obs):
        o = torch.as_tensor(np.asarray(obs, np.float32)[None], device=self.device)
        return float(self.critic(o).squeeze())

    def store(self, o, a, logp, r, done, v):
        self._buf["o"].append(o); self._buf["a"].append(a)
        self._buf["logp"].append(logp); self._buf["r"].append(r)
        self._buf["done"].append(float(done)); self._buf["v"].append(v)

    def buffer_full(self):
        return len(self._buf["o"]) >= self.rollout

    def update(self, last_val=0.0):
        b = self._buf
        o = torch.as_tensor(np.array(b["o"], np.float32), device=self.device)
        a = torch.as_tensor(np.array(b["a"], np.float32), device=self.device)
        logp_old = torch.as_tensor(np.array(b["logp"], np.float32), device=self.device)
        r = np.array(b["r"], np.float64)
        done = np.array(b["done"], np.float64)
        v = np.array(b["v"], np.float64)

        # GAE
        adv = np.zeros_like(r)
        gae = 0.0
        for t in reversed(range(len(r))):
            next_v = last_val if t == len(r) - 1 else v[t + 1]
            mask = 1.0 - done[t]
            delta = r[t] + self.gamma * next_v * mask - v[t]
            gae = delta + self.gamma * self.lam * mask * gae
            adv[t] = gae
        ret = adv + v
        adv_t = torch.as_tensor(np.asarray((adv - adv.mean()) / (adv.std() + 1e-8), np.float32), device=self.device)
        ret_t = torch.as_tensor(np.asarray(ret, np.float32), device=self.device)

        n = len(r)
        idx = np.arange(n)
        stats = []
        for _ in range(self.epochs):
            np.random.shuffle(idx)
            for s in range(0, n, self.minibatch):
                mb = idx[s:s + self.minibatch]
                if len(mb) < 2:
                    continue
                mb_t = torch.as_tensor(mb, device=self.device)
                mean, std = self._dist(o[mb_t])
                x = torch.atanh(torch.clamp(a[mb_t], -0.999, 0.999))
                logp = self._logp(mean, std, x) - torch.log(1 - a[mb_t].pow(2) + 1e-6).sum(-1)
                ratio = torch.exp(logp - logp_old[mb_t])
                surr1 = ratio * adv_t[mb_t]
                surr2 = torch.clamp(ratio, 1 - self.clip, 1 + self.clip) * adv_t[mb_t]
                p_loss = -torch.min(surr1, surr2).mean()
                ent = (0.5 * torch.log(2 * np.pi * std ** 2) + 0.5).sum(-1).mean()
                v_pred = self.critic(o[mb_t]).squeeze(-1)
                v_loss = nn.functional.mse_loss(v_pred, ret_t[mb_t])
                self.opt_a.zero_grad()
                (p_loss - self.ent_coef * ent).backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                self.opt_a.step()
                self.opt_c.zero_grad()
                (self.vf_coef * v_loss).backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                self.opt_c.step()
                stats.append((float(p_loss), float(v_loss), float(ent)))
        self._buf = {k: [] for k in ("o", "a", "logp", "r", "done", "v")}
        m = np.array(stats).mean(axis=0)
        return {"p_loss": m[0], "v_loss": m[1], "ent": m[2]}

    def save(self, path):
        torch.save({"actor": self.actor.state_dict(),
                    "log_std": self.log_std.detach().cpu(),
                    "critic": self.critic.state_dict()}, path)

    def load(self, path):
        ck = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(ck["actor"])
        with torch.no_grad():
            self.log_std.copy_(ck["log_std"].to(self.device))
        self.critic.load_state_dict(ck["critic"])
