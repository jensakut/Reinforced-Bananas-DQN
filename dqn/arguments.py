import argparse


def get_args():
    parse = argparse.ArgumentParser()
    # configuration of the RL Algorithm
    parse.add_argument('--use_double_q_learning', type=bool, default=True, help='use double dqn to train the agent')
    parse.add_argument('--use_dueling_q_learning', type=bool, default=True, help='use dueling to train the agent')
    parse.add_argument('--use_prioritized_experience_replay', type=bool, default=True,
                       help='use prioritized experience replay to train the agent')
    parse.add_argument('--use_fc_model', type=bool, default=True,
                       help='use prioritized experience replay to train the agent')

    # directories
    parse.add_argument('--save-dir', type=str, default='../saved_models/', help='the folder to save models')
    parse.add_argument('--log-dir', type=str, default='../logs/', help='dir to save log information')
    parse.add_argument('--sim_dir', type=str, default='../Banana_Linux/Banana.x86_64', help='simulator directory')
    parse.add_argument('--train', type=bool, default=True, help='Train')

    # network architecture
    parse.add_argument('--fc1', type=int, default=int(64), help='the fc1 network layers')
    parse.add_argument('--fc2', type=int, default=int(64), help='the fc2 network layers')

    # dqn common parameters mainly used in train.py
    parse.add_argument('--n_episodes', type=int, default=int(2000), help='the total timesteps to train network')
    parse.add_argument('--max_t', type=int, default=int(300), help='maximum number of timesteps per episode')
    parse.add_argument('--epsilon_start', type=float, default=1.0, help='the initial exploration ratio')
    parse.add_argument('--epsilon_min', type=float, default=0.01, help='the final exploration ratio')
    parse.add_argument('--epsilon_decay', type=float, default=0.99, help='the exploration decay')
    parse.add_argument('--gamma', type=float, default=0.99, help='the discount factor of RL (eps_decay)')

    # training arguments
    parse.add_argument('--seed', type=int, default=0, help='the random seeds')
    parse.add_argument('--batch_size', type=int, default=512, help='the batch size of updating')
    parse.add_argument('--lr', type=float, default=1e-4, help='learning rate of the algorithm')
    parse.add_argument('--buffer_size', type=int, default=int(2 ** 15),
                       help='the size of the replay buffer. An episode is'
                            '300 long. The tree is best with factor 2. ')
    parse.add_argument('--tau', type=float, default=1e-2, help='for soft update of target parameters')
    parse.add_argument('--update_every', type=int, default=16, help='how often to update the network')

    # prioritized replay buffer
    parse.add_argument('--per_max_priority', type=float, default=1.0, help='the maximum priority for an experience')
    parse.add_argument('--per_alpha', type=float, default=0.4, help='the weighting factor a')
    parse.add_argument('--per_alpha_end', type=float, default=0.4, help='the weighting factor a')
    parse.add_argument('--per_beta', type=float, default=0.4, help='the weighting factor b')
    parse.add_argument('--per_beta_end', type=float, default=1.0, help='the weighting factor b')
    parse.add_argument('--per_annihilation', type=float, default=800, help='the weighting factor b')
    parse.add_argument('--per_eps', type=float, default=float(1e-3), help='the priority offset added to each priority')

    args = parse.parse_args()

    return args
