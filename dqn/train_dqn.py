import time

import torch
from collections import deque
from unityagents import UnityEnvironment
import numpy as np
from Agent import Agent
from arguments import get_args
from plotting import Plotting



# dqn function training the agent

def dqn(agent, args, train=True):
    """Deep Q-Learning.

    Args
        args: defined in arguments.py
        train (bool): flag deciding if the agent will train or just play through the episode
    """

    ##############################
    # init
    #############################
    scores = []  # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    scores_mean = []
    best_score = -1000.0
    max_t = 0
    # have some plotting object that captures the scores
    plotting = Plotting()
    eps = args.eps_start  # initialize epsilon

    #############################
    # run the episodes
    #############################
    for i_episode in range(1, args.n_episodes + 1):
        env_info = env.reset(train_mode=train)[brain_name]
        state = env_info.vector_observations[0]
        score = 0
        max_score = 15
        t_s = time.time()
        for t in range(args.max_t):
            action = agent.act(state, eps if train else 0.0)
            env_info = env.step(action)[brain_name]
            next_state = env_info.vector_observations[0]  # get the next state
            reward = env_info.rewards[0]  # get the reward
            done = env_info.local_done[0]  # see if episode has finished
            if train:
                agent.step(state, action, reward, next_state, done)
            score += reward  # update the score
            state = next_state  # roll over the state to next time step
            if done:  # exit loop if episode finished
                eps = max(args.eps_end, args.eps_decay * eps)  # decrease epsilon
                ctime = time.time() - t_s
                max_t = t
                break

        scores.append(score)
        scores_window.append(score)
        scores_mean.append(np.mean(scores_window))
        best_avg_score=max(best_score, scores_mean[-1])
        best_score=max(best_score, score)
        plotting.add_measurement(score=score, mean_score=scores_mean[-1], episode_length=max_t, epsilon=eps,
                                 alpha=agent.alpha, beta=agent.beta)

        print('\rEpisode {}\tAverage Score: {:.2f}\tScore: {:.2f}, episode_dur {:.2f}, max_t {}'.format(
            i_episode, scores_mean[-1], score, ctime, max_t), end="")
        if i_episode % 20 == 0:
            plotting.plotting(args=args, id=i_episode)
            print('\rEpisode {}\tAverage Score: {:.2f}\tScore: {:.2f}, episode_dur {:.2f}, max_t {}'.format(
                i_episode, scores_mean[-1], score, ctime, max_t))
        if scores_mean[-1] > max_score and train:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode - 100,
                                                                                         np.mean(scores_window)))
            torch.save(agent.qnetwork_local.state_dict(), 'local_network_episode_' + str(i_episode) + '.pth')
            plotting.plotting(args=args, id=i_episode)
            break
    return scores, scores_mean


# main function
if __name__ == "__main__":


    # get arguments
    args = get_args()

    # initialize the environment
    env = UnityEnvironment(file_name=args.sim_dir)

    # get the default brain
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    # reset the environment
    env_info = env.reset(train_mode=True)[brain_name]

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
    scores_double, scores_mean_double = dqn(agent, args)
