from collections import namedtuple

import numpy as np
import torch

from .SumTree import SumTree

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# based on https://github.com/rlcode/per

class PrioritizedExperienceReplay:
    """Fixed-size buffer to store experience tuples."""
    """
    e = 0.01
    a = 0.6
    beta = 0.4
    beta_increment_per_sampling = 0.001
    """

    def __init__(self, action_size, args):
        """Initialize a ExperienceReplay object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.seed = args.seed
        self.max_priority = args.per_max_priority
        self.alpha = args.per_alpha
        self.beta = args.per_beta
        self.alpha_increment = (args.per_alpha_end - args.per_alpha) / args.per_annihilation
        self.beta_increment = (args.per_beta_end - args.per_beta) / args.per_annihilation
        self.eps = args.per_eps
        self.i_episode = 0  # set from agent
        self.max_priority = args.per_max_priority
        self.tree = SumTree(args.buffer_size)
        self.capacity = args.buffer_size
        self.batch_size = args.batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])

    def _get_priority(self, error):
        # Sample transition probability P(i) = p_i^alpha / sum_k (p_k^alpha)
        # instead, just safe priorities with the alpha squared already.
        return (np.abs(error) + self.eps) ** self.alpha

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        experience = self.experience(state, action, reward, next_state, done)
        p = self.max_priority
        self.tree.add(p, experience)

        # done signals episode has ended. Then change alpha and beta
        # alpha and beta outside 0...1 makes no sense
        if done:
            self.alpha = np.max(np.min([1., self.alpha + self.alpha_increment]), 0)
            self.beta = np.max(np.min([1., self.beta + self.beta_increment]), 0)

    def sample(self):
        """Prioritized randomly sample a batch of experiences from memory."""
        experiences = []
        idxs = []
        priorities = []

        # sample a batch of random experiences by drawing a float between 0 and the sum of the tree
        s = np.random.uniform(0, self.tree.total(), self.batch_size)
        for i in range(self.batch_size):
            (idx, p, experience) = self.tree.get(s[i])
            priorities.append(p)
            experiences.append(experience)
            idxs.append(idx)

        # pseudocode 10: compute importance-sampling weight w_j = (N + P(j))^-beta / max_i w_i
        sampling_probabilities = priorities / self.tree.total()
        is_weights = np.power(self.tree.n_entries * sampling_probabilities, -self.beta)
        is_weights /= is_weights.max()

        # put all the stuff on the device (gpu) using tensors.
        # if pytorch is not used, just return numpy arrays and uncomment these lines
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(
            device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(
            device)
        is_weights = torch.from_numpy(is_weights).float().to(device)
        return idxs, is_weights, states, actions, rewards, next_states, dones

    def update_priorities(self, idxs, errors):
        for idx, error in zip(idxs, errors):
            p = self._get_priority(error)
            self.tree.update(idx, p)

    def __len__(self):
        """Return the current size of internal memory."""
        return self.tree.n_entries
