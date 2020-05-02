import random
import torch
import numpy as np

from collections import namedtuple, deque


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class PrioritizedExperienceReplay:
    """Fixed-size buffer to store experience tuples."""
    # Source: https://arxiv.org/abs/1511.05952
    # pseudocode found on page 5 of the publication

    def __init__(self, action_size, args):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=args.buffer_size)
        self.batch_size = args.batch_size
        # PRB: Add priority
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state",
                                                                "priority", "done"])
        self.seed = random.seed(args.seed)
        # PRB
        self.priorities = deque(maxlen=args.buffer_size)
        self.max_priority = args.max_priority
        self.alpha = args.alpha
        self.beta = args.beta
        self.eps = args.eps

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""

        # Store transition with maximal priority (line 6 pseudocode)
        e = self.experience(state, action, reward, next_state, self.max_priority, done)
        self.memory.append(e)
        self.priorities.append(self.max_priority)

    def sample(self):
        """Prioritized randomly sample a batch of experiences from memory."""
        priorities = np.array(self.priorities)
        # Sample transition probability P(i) = p_i^alpha / sum_k (p_k^alpha)
        # pseudocode 9
        assert(min(priorities) > 0)
        probabilities = priorities ** self.alpha / sum(priorities ** self.alpha)
        sample_indices = np.random.choice(np.arange(len(self.memory)), size=self.batch_size, p=probabilities)
        experiences = []
        for i in sample_indices:
            experiences.append(self.memory[i])

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(
            device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(
            device)

        # pseudocode 10: compute importance-sampling weight w_j = (N + P(j))^-beta / max_i w_i
        is_weights = np.power(len(self.memory) * probabilities, -self.beta)
        is_weights /= max(is_weights)

        is_weights = torch.from_numpy(is_weights).float().to(device)

        return sample_indices, states, actions, rewards, next_states, is_weights, dones

    def update_priorities(self, memory_indizes, priorities):
        for i, p in zip(memory_indizes, priorities):
            p += self.eps
            state, action, reward, next_state, priority, done = self.memory[i]
            self.memory[i] = self.experience(state, action, reward, next_state, p, done)
            self.priorities[i] = p


    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
