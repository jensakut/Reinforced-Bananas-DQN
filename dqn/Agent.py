import numpy as np
import random

# from model import QNetwork
from ExperienceReplay import ExperienceReplay
from PrioritizedExperienceReplay import PrioritizedExperienceReplay
from QNetwork_fc import QNetwork_fc

import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Agent:
    """Interacts with and learns from the environment."""

    def __init__(self, args, state_size, action_size, filename):
        """Initialize an Agent object.

        Params
        ======
            use_double_q_learning (bool): if 'True' use double DQN agent
            use_dueling_q_learning (bool): if 'True' use dueling agent
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.use_double_q_learning = args.use_double_q_learning
        self.use_dueling_q_learning = args.use_dueling_q_learning
        self.use_prioritized_experience_replay = args.use_prioritized_experience_replay

        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(args.seed)
        self.batch_size = args.batch_size
        self.gamma = args.gamma
        self.tau = args.tau
        self.update_every = args.update_every
        self.use_prioritized_experience_replay = args.use_prioritized_experience_replay
        self.eps = args.eps

        self.alpha = args.alpha
        self.beta = args.beta
        self.d_alpha = (args.alpha_end - args.alpha) / args.ann_length
        self.d_beta = (args.beta_end - args.beta) / args.ann_length
        self.ann_length = args.ann_length

        # Q-Network

        self.qnetwork_local = QNetwork_fc(state_size, action_size, args.seed, fc1_units=64, fc2_units=64, use_dueling=self.use_dueling_q_learning).to(device)
        self.qnetwork_target = QNetwork_fc(state_size, action_size, args.seed, fc1_units=64, fc2_units=64, use_dueling=self.use_dueling_q_learning).to(device)

        print(self.qnetwork_local)

        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=args.lr)

        self.mse_loss = nn.MSELoss()
        self.mse_element_loss = nn.MSELoss(reduce=False)
        self.l1_loss = nn.L1Loss(reduce=False)

        if filename:
            weights = torch.load(filename)
            self.qnetwork_local.load_state_dict(weights)
            self.qnetwork_target.load_state_dict(weights)

        # Replay memory
        if args.use_prioritized_experience_replay:
            self.memory = PrioritizedExperienceReplay(action_size, args)
        else:
            self.memory = ExperienceReplay(action_size, args.buffer_size, args.batch_size, args.seed)

        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
        self.i_episode = 1



    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)

        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > self.batch_size:
                experiences = self.memory.sample()
                self.learn(experiences, self.gamma)

        if done and self.use_prioritized_experience_replay and self.i_episode < self.ann_length:
            self.alpha += self.d_alpha
            self.beta += self.d_beta
            self.i_episode += 1

    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.

        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        if self.use_prioritized_experience_replay:
            sample_indices, states, actions, rewards, next_states, is_weights, dones = experiences
        else:
            states, actions, rewards, next_states, dones = experiences

        # Get max predicted Q values (for next states) from target model
        if self.use_double_q_learning:
            indices = torch.argmax(self.qnetwork_local(next_states).detach(), 1)
            q_targets_next = self.qnetwork_target(next_states).detach().gather(1, indices.unsqueeze(1))
        else:
            # Get max predicted Q values (for next states) from target model
            q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)

        # Compute Q targets for current states
        q_targets = rewards + (gamma * q_targets_next * (1 - dones))

        # Get expected Q values from local model
        q_expected = self.qnetwork_local(states).gather(1, actions)

        # Compute loss
        # TODO: Assess self.criterion = nn.SmoothL1Loss()
        if self.use_prioritized_experience_replay:
            # pseudocode 11, loss is td_error and pseudocode 12 priorities is abs td_error
            priorities = abs(q_targets - q_expected)
            # pseudocode 13
            loss = (is_weights * self.mse_element_loss(q_targets, q_expected)).mean()
            # Update Priorities based on offseted TD error
            self.memory.update_priorities(sample_indices, priorities.squeeze().to('cpu').data.numpy())
        else:
            loss = F.mse_loss(q_expected, q_targets)

        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target)

    def soft_update(self, local_model, target_model):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)

