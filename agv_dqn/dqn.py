"""DQN 智能体: Q 网络、经验回放、目标网络、epsilon-greedy 探索。

核心公式:
    TD 目标   y = r + gamma * max_a' Q(s', a'; theta^-) * (1 - done)
    网络损失  L(theta) = E[(y - Q(s, a; theta))^2]   (实现中用 Huber/SmoothL1)
    目标网络  theta^- <- theta   (每 target_update 次学习硬更新一次)
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class QNetwork(nn.Module):
    """全连接 Q 网络, 输入状态向量, 输出各动作的 Q 值。"""

    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ReplayBuffer:
    """经验回放池 D = {(s, a, r, s', done)}。"""

    def __init__(self, capacity: int, obs_dim: int) -> None:
        self.capacity = int(capacity)
        self.obs = np.zeros((self.capacity, obs_dim), dtype=np.float32)
        self.act = np.zeros(self.capacity, dtype=np.int64)
        self.rew = np.zeros(self.capacity, dtype=np.float32)
        self.nxt = np.zeros((self.capacity, obs_dim), dtype=np.float32)
        self.done = np.zeros(self.capacity, dtype=np.float32)
        self.ptr = 0
        self.size = 0

    def push(self, obs, act, rew, nxt, done) -> None:
        i = self.ptr
        self.obs[i] = obs
        self.act[i] = int(act)
        self.rew[i] = float(rew)
        self.nxt[i] = nxt
        self.done[i] = float(done)
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, device: torch.device):
        idx = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.as_tensor(self.obs[idx], device=device),
            torch.as_tensor(self.act[idx], device=device),
            torch.as_tensor(self.rew[idx], device=device),
            torch.as_tensor(self.nxt[idx], device=device),
            torch.as_tensor(self.done[idx], device=device),
        )

    def __len__(self) -> int:
        return self.size


class DQNAgent:
    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        hidden: int = 128,
        lr: float = 1e-3,
        gamma: float = 0.99,
        buffer_capacity: int = 50_000,
        batch_size: int = 64,
        target_update: int = 200,
        eps_start: float = 1.0,
        eps_end: float = 0.05,
        eps_decay_episodes: int = 250,
        double_dqn: bool = False,
        seed: int = 0,
        device: torch.device | None = None,
    ) -> None:
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.n_actions = int(n_actions)
        self.gamma = float(gamma)
        self.batch_size = int(batch_size)
        self.target_update = int(target_update)
        self.double_dqn = bool(double_dqn)
        self.eps_start = float(eps_start)
        self.eps_end = float(eps_end)
        self.eps_decay_episodes = int(eps_decay_episodes)

        torch.manual_seed(seed)
        np.random.seed(seed)
        self.rng = np.random.default_rng(seed)

        self.q = QNetwork(obs_dim, n_actions, hidden).to(self.device)
        self.q_target = QNetwork(obs_dim, n_actions, hidden).to(self.device)
        self.q_target.load_state_dict(self.q.state_dict())
        self.q_target.eval()
        self.optimizer = torch.optim.Adam(self.q.parameters(), lr=lr)
        self.buffer = ReplayBuffer(buffer_capacity, obs_dim)
        self.learn_steps = 0
        self.episode = 0

    def epsilon(self) -> float:
        """线性衰减的探索率。"""
        frac = min(1.0, self.episode / max(1, self.eps_decay_episodes))
        return self.eps_start + (self.eps_end - self.eps_start) * frac

    @torch.no_grad()
    def act(self, obs: np.ndarray, greedy: bool = False) -> int:
        if not greedy and self.rng.random() < self.epsilon():
            return int(self.rng.integers(self.n_actions))
        x = torch.as_tensor(np.asarray(obs, dtype=np.float32)[None], device=self.device)
        return int(self.q(x).argmax(dim=1).item())

    @torch.no_grad()
    def q_values(self, obs: np.ndarray) -> np.ndarray:
        x = torch.as_tensor(np.asarray(obs, dtype=np.float32)[None], device=self.device)
        return self.q(x).squeeze(0).cpu().numpy()

    def observe(self, obs, act, rew, nxt, terminated) -> None:
        # 注意: 存 terminated 而非 truncated, 截断时仍需 bootstrap
        self.buffer.push(obs, act, rew, nxt, terminated)

    def learn(self) -> float | None:
        if len(self.buffer) < self.batch_size:
            return None
        obs, act, rew, nxt, done = self.buffer.sample(self.batch_size, self.device)
        q_pred = self.q(obs).gather(1, act.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            if self.double_dqn:
                a_next = self.q(nxt).argmax(dim=1, keepdim=True)
                q_next = self.q_target(nxt).gather(1, a_next).squeeze(1)
            else:
                q_next = self.q_target(nxt).max(dim=1).values
            target = rew + self.gamma * q_next * (1.0 - done)
        loss = nn.functional.smooth_l1_loss(q_pred, target)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q.parameters(), 10.0)
        self.optimizer.step()

        self.learn_steps += 1
        if self.learn_steps % self.target_update == 0:
            self.q_target.load_state_dict(self.q.state_dict())
        return float(loss.item())

    def end_episode(self) -> None:
        self.episode += 1

    def save(self, path) -> None:
        torch.save(
            {
                "q": self.q.state_dict(),
                "q_target": self.q_target.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "episode": self.episode,
            },
            path,
        )

    def load(self, path) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.q.load_state_dict(ckpt["q"])
        self.q_target.load_state_dict(ckpt["q_target"])
        self.q.eval()
        self.q_target.eval()
