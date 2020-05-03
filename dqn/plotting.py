import time

import matplotlib.pyplot as plt
import numpy as np


class Plotting:
    def __init__(self):
        self.scores = []
        self.scores_mean = []
        self.lower = []
        self.upper = []
        self.scores_min = []
        self.scores_max = []
        self.episode_length = []
        self.max_episode_length = 0
        self.epsilon = []
        self.eps_x = []
        self.eps_y = []
        self.eps_next = 1
        self.target = []
        self.score_int = 0
        self.count = 0
        self.score_ints_x = []
        self.score_ints = []
        self.timestamp = time.time()
        self.alpha = []
        self.beta = []

    def add_measurement(self, score, mean_score, episode_length, epsilon, alpha, beta):
        self.scores.append(score)
        self.episode_length.append(episode_length)
        self.max_episode_length = max(self.max_episode_length, episode_length)
        self.epsilon.append(epsilon)
        if epsilon <= self.eps_next:
            self.eps_next *= 0.5
            self.eps_x.append(self.count)
            self.eps_y.append(epsilon)
        if mean_score >= self.score_int + 1:
            self.score_int += 1
            self.score_ints_x.append(self.count)
            self.score_ints.append(self.score_int)
        self.scores_mean.append(mean_score)
        std = np.std(self.scores)
        mean = self.scores_mean[-1]
        self.lower.append(mean - std)
        self.upper.append(mean + std)
        self.target.append(13)
        self.alpha.append(alpha)
        self.beta.append(beta)
        self.count += 1

    # do some logging and plotting
    def plotting(self, args, id):
        # plot the scores
        # fig = plt.figure(num=id)
        fig, axs = plt.subplots(2, 1, constrained_layout=True, num=id, dpi=500)
        axs[0].plot(np.arange(len(self.scores)), self.scores, label='score')
        axs[0].plot(np.arange(len(self.scores_mean)), self.scores_mean, label='mean score')
        axs[0].plot(np.arange(len(self.lower)), self.lower, label='mean+std')
        axs[0].plot(np.arange(len(self.upper)), self.upper, label='mean-std')
        axs[0].plot(np.arange(len(self.target)), self.target, label='target')
        axs[0].plot(self.score_ints_x, self.score_ints, '.', label='mean score int')

        # axs[0].plot(np.arange(len(self.scores_min)), self.scores_min, label='100 min score')
        # axs[0].plot(np.arange(len(self.scores_max)), self.scores_max, label='100 max score')
        axs[0].legend()
        axs[0].set_ylabel('Score')

        axs[1].plot(np.arange(len(self.epsilon)), self.epsilon, label='epsilon')
        axs[1].plot(self.eps_x, self.eps_y, '.', label='eps halftime')
        axs[1].plot(np.arange(len(self.alpha)), self.alpha, label='alpha')
        axs[1].plot(np.arange(len(self.beta)), self.beta, label='beta')
        axs[1].plot(np.arange(len(self.episode_length)), np.array(self.episode_length) / self.max_episode_length,
                    label='Norm. episode_length')

        axs[1].set_ylabel('epsilon')
        axs[1].legend()

        axs[1].set_xlabel('Episode Number')
        name = "double_{}_duel_{}_per_{}_lr_{}_gamma_{}_batch_{}_tau_{}_alpha_{:.1}_beta_{:.1}_per.eps_{:.1}_updint_{}_eps.decay_{}_end_{}_{}.png".format(
            args.use_double_q_learning, args.use_dueling_q_learning, args.use_prioritized_experience_replay, args.lr,
            args.gamma, args.batch_size, args.tau, args.alpha, args.beta, args.eps, args.update_every, args.eps_decay,
            args.eps_end, self.timestamp)
        plt.savefig(name)
        plt.close(id)

    def write_csv(self, args, csv, best_score, scores):
        list = [args.use_double_q_learning, args.use_dueling_q_learning, args.use_prioritized_experience_replay,
                args.lr,
                args.gamma, args.batch_size, args.tau, args.alpha, args.beta, args.eps, args.update_every,
                args.eps_decay,
                args.eps_end, self.timestamp, best_score, scores, alpha, beta, epsilon]
        with open('innovators.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["SN", "Name", "Contribution"])
