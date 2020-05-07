import random
from copy import copy

import numpy as np


def policy_to_args(policy, args):
    [args.lr, args.fc1, args.fc2, args.per_annihilation, args.per_alpha, args.per_beta,
     args.batch_size, args.epsilon_decay, args.tau, args.update_every, args.per_eps, args.buffer_size] = policy
    args.per_alpha_end = args.per_alpha
    return args


def get_policy(args):
    return [args.lr, args.fc1, args.fc2, args.per_annihilation, args.per_alpha, args.per_beta,
            args.batch_size, args.epsilon_decay, args.tau, args.update_every, args.per_eps, args.buffer_size]


def _optimization_formula(rewards):
    R_max = np.max(rewards[1])
    R_mean = np.mean(rewards[0]) / len(rewards[0])
    print("R_max {} R_mean {}".format(R_max, R_mean))
    return 0.75 * R_max + 0.25 * R_mean


class PolicySearch:

    def __init__(self, args):
        self.best_R = -np.inf
        self.best_policy = get_policy(args)
        self.policies = []
        self.rewards = []
        self.Rs = []
        self.iteration = 0

        learning_rates = [1e-3, 5e-4, 1e-4, 5e-5]
        fc1s = [64, 128, 256, 512]
        fc2s = [32, 64, 128, 256, 512]
        ann_lengths = [0.25 * args.n_episodes, 0.5 * args.n_episodes, 0.75 * args.n_episodes, args.n_episodes]
        # alpha per will stay constant
        alphas = [0.4, 0.6, 0.8, 1.0]
        # beta will be annihilated to 1.0 with ann
        betas = [0.4, 0.6, 0.8, 1.0]
        batch_sizes = [32, 64, 128, 256, 512, 1024]
        epsilon_decays = [0.9999, 0.995, 0.992, 0.99, 0.985, 0.98]
        taus = [1e-2, 5e-2, 1e-3]
        update_everys = [8, 16, 32, 64]
        per_epss = [1e-3, 1e-2]
        buffer_sizes = [2 ** 14, 2 ** 15, 2 ** 16]

        self.parameters = [learning_rates, fc1s, fc2s, ann_lengths, alphas, betas,
                           batch_sizes, epsilon_decays, taus, update_everys, per_epss, buffer_sizes]
        self.name_parameters = ['learning_rate', 'fc1', 'fc2', 'annihilation_lengths', 'alpha', 'beta_start',
                                'batch_size', 'epsilon_decay', 'tau', 'update_every', 'per_eps', 'buffer_size']
        assert len(self.parameters) is len(self.name_parameters)
        self.n_parameters = len(self.parameters)

    def _hill_climbing(self):
        # return the policy last in the list (highest score)
        # policy is second element in the list
        return copy(self.policies[-1][1])

    def generate_policy(self, args, rewards):
        print("")
        print("generate new policy {}".format(self.iteration))
        p = get_policy(args)
        print("{} is the previously tested policy".format(p))
        R = _optimization_formula(rewards)
        self.policies.append([R, p])
        self.policies.sort()
        print("policy history: ")
        for policy in self.policies:
            print("{} policy; {:.1f} R".format(policy[1], policy[0]))
        next_policy = self._hill_climbing()
        print("{} is the base policy for mutation. ".format(next_policy))

        mutated_policy = self.mutate_policy(next_policy)
        args = policy_to_args(mutated_policy, args)

        self.iteration += 1
        return args

    def mutate_policy(self, next_policy):
        # mutate new_policy by randomly flipping one or two parameters
        # while loop prevents flipping and randomly keeping the same policy
        criterion = True
        while criterion:
            # randomly change parameters
            change_ids = [random.randint(0, self.n_parameters - 1) for i in range(2)]
            for change_id in change_ids:
                # take new parameters
                mutated_parameter_id = random.randint(0, len(self.parameters[change_id]) - 1)
                next_policy[change_id] = self.parameters[change_id][mutated_parameter_id]
                print("mutate parameter {} to {}".format(self.name_parameters[change_id],
                                                         self.parameters[change_id][mutated_parameter_id]))
            criterion = False
            for policy in self.policies:
                if next_policy == policy[1]:
                    criterion = True
                    print("{} to be tested matches the previous policy {} with Reward {}".format(next_policy, policy[1],
                                                                                                 policy[0]))
        print("{} is the mutated policy which will be tested next".format(next_policy, criterion))
        return next_policy
