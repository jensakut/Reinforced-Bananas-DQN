import time
from collections import deque

import numpy as np
import torch
from Agent import Agent
from arguments import get_args
from plotting import Plotting
from unityagents import UnityEnvironment


# dqn function training the agent

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
    scores_mean = []
    best_score = -1000.0
    best_avg_score = -1000.0
    max_t = 0
    # have some plotting object that captures the scores
    plotting = Plotting()
    timeout_score = 0

    #############################
    # run the episodes
    #############################
    for i_episode in range(1, args.n_episodes + 1):
        env_info = env.reset(train_mode=args.train)[brain_name]
        state = env_info.vector_observations[0]
        score = 0
        max_score = 16
        t_s = time.time()
        for t in range(args.max_t):
            action = agent.act(state)
            env_info = env.step(action)[brain_name]
            next_state = env_info.vector_observations[0]  # get the next state
            reward = env_info.rewards[0]  # get the reward
            done = env_info.local_done[0]  # see if episode has finished
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
        plotting.add_measurement(score=score, mean_score=scores_mean[-1], episode_length=max_t, epsilon=agent.eps,
                                 alpha=agent.memory.alpha, beta=agent.memory.beta)

        best_score = max(score, best_score)
        # check whether it is still improving at least a bit
        if best_avg_score < scores_mean[-1]:
            best_avg_score = scores_mean[-1]
            timeout_score = 0
        # at the beginning it is particularily noisy so deactivate the timeout a bit
        if i_episode < 0.05 * args.n_episodes:
            timeout_score = 0
        # if nothing improving the score is learned after 100 episodes stop

        elif timeout_score > 100:
            plotting.plotting(args=args, id=i_episode)
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
            torch.save(agent.qnetwork_local.state_dict(),
                       'local_network_episode_' + str(i_episode) + '_score_' + str(scores_mean[-1]) + '.pth')

        # save a network
        if scores_mean[-1] > max_score and args.train:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode,
                                                                                         np.mean(scores_window)))
            torch.save(agent.qnetwork_local.state_dict(),
                       'local_network_episode_' + str(i_episode) + '_score_' + str(scores_mean[-1]) + '.pth')
            plotting.plotting(args=args, id=i_episode)

    return best_avg_score, scores


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
    scores_double, scores_mean_double = dqn(agent, args)
