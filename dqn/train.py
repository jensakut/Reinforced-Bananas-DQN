import time
from collections import deque

import numpy as np
import torch

from dqn.Agent import Agent
from dqn.UnityWrapper import UnityWrapper
from dqn.arguments import get_args
from dqn.plotting import Plotting


# dqn function training the agent

def dqn(agent, args, env):
    """Deep Q-Learning.

    Args
        args: defined in arguments.py
    """

    ##############################
    # init
    #############################
    scores = []  # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    scores_mean = []
    best_score = -1000.0
    best_avg_score = -1000.0
    max_t = 0
    # have some plotting object that captures the scores
    plotting = Plotting(args)
    timeout_score = 0
    max_downswing = 3.
    downswing = 0.
    max_score = 18

    #############################
    # run the episodes
    #############################
    for i_episode in range(1, args.n_episodes + 1):
        state = env.reset()
        score = 0
        t_s = time.time()
        for t in range(args.max_t):
            action = agent.act(state)
            reward, next_state, done = env.step(action)  # get the next state
            if args.train:
                agent.step(state, action, reward, next_state, done)
            score += reward  # update the score
            state = next_state  # roll over the state to next time step
            if done:  # exit loop if episode finished
                ctime = time.time() - t_s
                max_t = t
                timeout_score += 1
                break

        # do the bookkeeping
        scores.append(score)
        scores_window.append(score)
        scores_mean.append(np.mean(scores_window))

        # analyze how much downswing there is
        # allow a maximum reduction of 2 in the average score
        if i_episode > 1:
            delta_m_score = scores_mean[-1] - scores_mean[-2]
            downswing = min(0, downswing + delta_m_score)

        plotting.add_measurement(score=score, mean_score=scores_mean[-1], episode_length=max_t, epsilon=agent.eps,
                                 alpha=agent.memory.alpha, beta=agent.memory.beta, downswing=downswing)

        best_score = max(score, best_score)
        # check whether it is still improving at least a bit
        if best_avg_score < scores_mean[-1]:
            best_avg_score = scores_mean[-1]
            timeout_score = 0
        # if nothing improving the score is learned after n episodes stop
        elif timeout_score > 300 and i_episode > 0.1 * args.n_episodes:
            plotting.plotting(args=args, id=i_episode)
            print("timeout of {:.1f} ended this training.".format(timeout_score))
            break
        # if score went down significantly, stop
        elif downswing < -max_downswing and i_episode > 0.05 * args.n_episodes:
            plotting.plotting(args=args, id=i_episode)
            print("downswing of magnitude {:.1f} ended this training.".format(downswing))
            break

        # state where we are
        print('\rEpisode {}\tAverage Score: {:.1f}\tScore: {:.0f}, episode_dur {:.2f}, best_avg_score {:.1f}, '
              'best_score {:.1f}, timeout_score {}'.format(i_episode, scores_mean[-1], score, ctime, best_avg_score,
                                                           best_score, timeout_score), end="")

        # save an image every 20 steps to allow for sneak peeking
        if i_episode % 20 == 0:
            plotting.plotting(args=args, id=i_episode)
            print('\rEpisode {}\tAverage Score: {:.2f}\tScore: {:.1f}, episode_dur {:.2f}'.format(
                i_episode, scores_mean[-1], score, ctime))
            if score > 0.9 * max_score:
                torch.save(agent.qnetwork_local.state_dict(),
                           args.save_dir + 'local_network_score_' + str(scores_mean[-1]) + '_i_episode_' + str(
                               i_episode) +
                           '_' + str(plotting.timestamp) + '.pth')

        # save a network
        if scores_mean[-1] > max_score and args.train:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode,
                                                                                         np.mean(scores_window)))
            torch.save(agent.qnetwork_local.state_dict(),
                       args.save_dir + 'local_network_score_' + str(scores_mean[-1]) + '_i_episode_' + str(i_episode) +
                       '_' + plotting.timestamp + '.pth')
            plotting.plotting(args=args, id=i_episode)

    return scores, scores_mean


# main function
if __name__ == "__main__":
    # get arguments
    args = get_args()

    env_wrapper = UnityWrapper(args)
    state_size, action_size, _ = env_wrapper.get_env_info()
    agent = Agent(args,
                  state_size=state_size,
                  action_size=action_size,
                  filename='')
    scores, scores_mean = dqn(agent, args, env_wrapper)
