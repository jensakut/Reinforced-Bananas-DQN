from collections import deque

import numpy as np
import torch
# get arguments
from unityagents import UnityEnvironment

from .Agent import Agent
from .arguments import get_args


def dqn(agent, args):
    """Deep Q-Learning.

    Args
        args: defined in arguments.py
    """

    ##############################
    # init
    #############################
    scores = []  # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores

    # have some plotting object that captures the scores
    eps = args.eps_start  # initialize epsilon

    #############################
    # run the episodes
    #############################
    for i_episode in range(1, args.n_episodes + 1):
        env_info = env.reset(train_mode=args.train)[brain_name]
        state = env_info.vector_observations[0]
        score = 0
        for t in range(args.max_t):
            # always greedy, show the best :-)
            action = agent.act(state, 0.0)
            env_info = env.step(action)[brain_name]
            next_state = env_info.vector_observations[0]  # get the next state
            reward = env_info.rewards[0]  # get the reward
            done = env_info.local_done[0]  # see if episode has finished
            score += reward  # update the score
            state = next_state  # roll over the state to next time step
            if done:  # exit loop if episode finished
                eps = max(args.eps_end, args.eps_decay * eps)  # decrease epsilon
                scores_window.append(score)
                break
    print('\rEpisode {}\tAverage Score: {:.2f}\tScore: {:.2f}'.format(i_episode, np.mean(scores_window), score), end="")


args = get_args()
args.train = False
args.n_episodes = 100

# initialize the environment
env = UnityEnvironment(file_name=args.sim_dir)

# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

# reset the environment
env_info = env.reset(train_mode=args.train)[brain_name]

# number of agents in the environment
print('Number of agents:', len(env_info.agents))

# number of actions
action_size = brain.vector_action_space_size
print('Number of actions:', action_size)

# examine the state space
state = env_info.vector_observations[0]
print('States look like:', state)
state_size = len(state)
print('States have length:', state_size)

agent = Agent(args,
              state_size=state_size,
              action_size=action_size,
              filename='')
# load best weights
agent.qnetwork_local.load_state_dict(torch.load('local_network_best_15_400_episodes.pth'))

dqn(agent, args)
