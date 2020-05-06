import csv
import time

import matplotlib.pyplot as plt
import numpy as np
import yaml


class Plotting:
    def __init__(self, args):
        self.scores = []
        self.scores_mean = []
        self.scores_mean_max = -np.inf
        self.scores_mean_max_idx = 0
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
        self.downswing = []
        self.name = "{}".format(self.timestamp)
        self.fname = args.log_dir + self.name
        self.dict_file = None

    def reset(self, args):
        self.__init__(args)

    def add_measurement(self, score, mean_score, episode_length, epsilon, alpha, beta, downswing):
        if mean_score > self.scores_mean_max:
            self.scores_mean_max = mean_score
            self.scores_mean_max_idx = self.count
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
        self.downswing.append(downswing)
        self.count += 1

    # do some logging and plotting
    def plotting(self, args, id):

        self._write_yaml(args)
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
        axs[1].plot(np.arange(len(self.downswing)), self.downswing, label='downswing')

        axs[1].set_ylabel('-')
        axs[1].legend()

        axs[1].set_xlabel('Episode Number')

        plt.savefig(self.fname + '.png')
        plt.close(id)

    def _write_yaml(self, args):
        dict_file = [
            {'max_score': float(self.scores_mean_max)},
            {'at_iteration': int(self.scores_mean_max_idx)},
            {'number_episodes_trained': self.count},
            {'experiment_name': self.name},
            {'double_dqn': args.use_double_q_learning},
            {'dueling_dqn': args.use_dueling_q_learning},
            {'per_dqn': args.use_prioritized_experience_replay},
            {'fc1': args.fc1},
            {'fc2': args.fc2},
            {'number_episodes': args.n_episodes},
            {'number_steps': args.max_t},
            {'epsilon_start': args.epsilon_start},
            {'epsilon_min': args.epsilon_min},
            {'epsilon_decay': args.epsilon_decay},
            {'gamma': args.gamma},
            {'seed': args.seed},
            {'batch_size': args.batch_size},
            {'learning_rate': args.lr},
            {'buffer_size': args.buffer_size},
            {'tau': args.tau},
            {'learn_every': args.update_every},
            {'per_max_priority': args.per_max_priority},
            {'per_alpha': args.per_alpha},
            {'per_alpha_end': args.per_alpha_end},
            {'per_beta': args.per_beta},
            {'per_beta_end': args.per_beta_end},
            {'per_epsilon': args.per_eps},
            {'timestamp': self.timestamp},
            {'per_annihilation': args.per_annihilation},
        ]
        yaml_name = self.fname + '.yaml'
        with open(yaml_name, 'w+') as yaml_file:
            yaml.dump_all([dict_file, list(self.scores_mean), list(self.scores), self.epsilon, self.alpha, self.beta],
                          yaml_file)

        csv_name = self.fname + '.csv'
        with open(csv_name, 'w', newline='') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(dict_file)
            writer.writerow(self.scores_mean)
            writer.writerow(self.scores)
            writer.writerow(self.epsilon)
            writer.writerow(self.alpha)
            writer.writerow(self.beta)
