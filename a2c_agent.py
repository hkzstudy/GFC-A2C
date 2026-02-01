# a2c_agent.py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Bernoulli
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from replay_buffer import ReplayBuffer

class Actor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, action_dim)
        )
    def forward(self, x): return self.net(x)

class Critic(nn.Module):
    def __init__(self, state_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1)
        )
    def forward(self, x): return self.net(x).squeeze(-1)

@dataclass
class Transition:
    state: torch.Tensor
    action: torch.Tensor
    reward: float
    done: bool
    log_prob: torch.Tensor
    value: torch.Tensor
    next_state: torch.Tensor

class A2CAgent:
    def __init__(self,
        state_dim: int, action_dim: int, gamma: float = 0.99,
        lr_actor: float = 3e-4, lr_critic: float = 3e-4,
        entropy_coef: float = 0.02, value_coef: float = 0.5,
        max_grad_norm: float = 0.5, device: str = "cpu",
        logits_clip: float = 10.0, normalize_advantages: bool = True,
        replay_capacity: int = 50000, critic_replay_updates: int = 4,
        replay_batch_size: int = 128, min_replay_size: int = 1000):
        self.device = torch.device(device)
        self.actor  = Actor(state_dim, action_dim).to(self.device)
        self.critic = Critic(state_dim).to(self.device)
        self.opt_actor  = optim.Adam(self.actor.parameters(),  lr=lr_actor)
        self.opt_critic = optim.Adam(self.critic.parameters(), lr=lr_critic)

        self.gamma = gamma
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.logits_clip = logits_clip
        self.normalize_advantages = normalize_advantages

        self.memory: List[Transition] = []
        self.replay = ReplayBuffer(capacity=replay_capacity, device=device)
        self.critic_replay_updates = critic_replay_updates
        self.replay_batch_size = replay_batch_size
        self.min_replay_size = min_replay_size
        self.mse = nn.MSELoss()

    @torch.no_grad()
    def select_action(self, state_np, mask_np=None, greedy=False):
        s = torch.as_tensor(state_np, dtype=torch.float32, device=self.device).unsqueeze(0)
        logits = self.actor(s).squeeze(0)
        if self.logits_clip is not None:
            logits = torch.clamp(logits, -self.logits_clip, self.logits_clip)
        if mask_np is not None:
            mask = torch.as_tensor(mask_np, dtype=torch.float32, device=self.device)
            logits = logits + (mask - 1.0) * 1e9
        if greedy:
            probs = torch.sigmoid(logits)
            a = (probs > 0.5).float()
            return a.cpu().numpy().astype(int), torch.tensor(0.0, device=self.device)
        dist = Bernoulli(logits=logits)
        a = dist.sample()
        logp = dist.log_prob(a).sum()
        return a.cpu().numpy().astype(int), logp

    @torch.no_grad()
    def value(self, state_np):
        s = torch.as_tensor(state_np, dtype=torch.float32, device=self.device).unsqueeze(0)
        return self.critic(s).item()

    def store(self, state, action, reward, done, log_prob, value, next_state):
        self.memory.append(Transition(
            state=torch.as_tensor(state, dtype=torch.float32),
            action=torch.as_tensor(action, dtype=torch.float32),
            reward=float(reward), done=bool(done),
            log_prob=log_prob.detach().cpu(),
            value=torch.as_tensor(value, dtype=torch.float32),
            next_state=torch.as_tensor(next_state, dtype=torch.float32),
        ))
        self.replay.add(state, action, reward, done, next_state)

    def _actor_critic_onpolicy_update(self) -> Dict[str, Any]:
        states      = torch.stack([t.state for t in self.memory]).to(self.device)
        actions     = torch.stack([t.action for t in self.memory]).to(self.device)
        rewards     = torch.as_tensor([t.reward for t in self.memory], device=self.device)
        dones       = torch.as_tensor([t.done   for t in self.memory], device=self.device)
        values_old  = torch.stack([t.value     for t in self.memory]).to(self.device)
        next_states = torch.stack([t.next_state for t in self.memory]).to(self.device)

        with torch.no_grad():
            next_values = self.critic(next_states)
            targets = rewards + self.gamma * (1.0 - dones.float()) * next_values
            adv = targets - values_old
            if self.normalize_advantages:
                adv = (adv - adv.mean()) / (adv.std(unbiased=False).clamp_min(1e-6))

        logits = self.actor(states)
        if self.logits_clip is not None:
            logits = torch.clamp(logits, -self.logits_clip, self.logits_clip)
        dist = Bernoulli(logits=logits)
        log_probs = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)

        actor_loss  = -(log_probs * adv.detach()).mean() - self.entropy_coef * entropy.mean()
        values = self.critic(states)
        critic_loss = self.mse(values, targets) * self.value_coef

        self.opt_actor.zero_grad();  actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        self.opt_actor.step()

        self.opt_critic.zero_grad(); critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
        self.opt_critic.step()

        return {"actor_loss": float(actor_loss.item()),
                "critic_loss": float(critic_loss.item()),
                "entropy": float(entropy.mean().item())}

    def _critic_replay_updates(self) -> Dict[str, Any]:
        if len(self.replay) < self.min_replay_size:
            return {"replay_steps": 0, "replay_critic_loss": 0.0}
        total, steps = 0.0, 0
        for _ in range(self.critic_replay_updates):
            if len(self.replay) < self.replay_batch_size: break
            states, rewards, dones, next_states = self.replay.sample(self.replay_batch_size)
            with torch.no_grad():
                targets = rewards + self.gamma * (1.0 - dones) * self.critic(next_states)
            values = self.critic(states)
            loss = self.mse(values, targets) * self.value_coef
            self.opt_critic.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            self.opt_critic.step()
            total += float(loss.item()); steps += 1
        return {"replay_steps": steps, "replay_critic_loss": (total/steps if steps>0 else 0.0)}

    def update(self):
        if not self.memory: return None
        stats1 = self._actor_critic_onpolicy_update()
        stats2 = self._critic_replay_updates()
        self.memory.clear()
        return {**stats1, **stats2}

    @torch.no_grad()
    def greedy_action(self, state_np, mask_np=None):
        a, _ = self.select_action(state_np, mask_np=mask_np, greedy=True)
        return a
