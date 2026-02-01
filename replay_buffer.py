# replay_buffer.py
import random
from dataclasses import dataclass
import torch

@dataclass
class ReplayItem:
    state: torch.Tensor
    action: torch.Tensor
    reward: float
    done: bool
    next_state: torch.Tensor

class ReplayBuffer:
    def __init__(self, capacity=50000, device="cpu"):
        self.capacity = int(capacity)
        self.storage = []
        self.idx = 0
        self.device = torch.device(device)

    def __len__(self): return len(self.storage)

    def add(self, state, action, reward, done, next_state):
        item = ReplayItem(
            state=torch.as_tensor(state, dtype=torch.float32),
            action=torch.as_tensor(action, dtype=torch.float32),
            reward=float(reward), done=bool(done),
            next_state=torch.as_tensor(next_state, dtype=torch.float32),
        )
        if len(self.storage) < self.capacity:
            self.storage.append(item)
        else:
            self.storage[self.idx] = item
        self.idx = (self.idx + 1) % self.capacity

    def sample(self, batch_size: int):
        batch = random.sample(self.storage, batch_size)
        states = torch.stack([b.state for b in batch]).to(self.device)
        rewards = torch.as_tensor([b.reward for b in batch], dtype=torch.float32, device=self.device)
        dones   = torch.as_tensor([b.done   for b in batch], dtype=torch.float32, device=self.device)
        next_states = torch.stack([b.next_state for b in batch]).to(self.device)
        return states, rewards, dones, next_states
