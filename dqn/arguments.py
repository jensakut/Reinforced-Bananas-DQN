import argparse

def get_args():
    parse = argparse.ArgumentParser()
    # configuration of the RL Algorithm
    parse.add_argument('--use_double_q_learning', type=bool, default=True, help='use double dqn to train the agent')
    parse.add_argument('--use_dueling_q_learning', type=bool, default=True, help='use dueling to train the agent')
    parse.add_argument('--use_prioritized_experience_replay', type=bool, default=True, help='use prioritized experience replay to train the agent')
    parse.add_argument('--use_fc_model', type=bool, default=True, help='use prioritized experience replay to train the agent')

    # directories
    parse.add_argument('--save-dir', type=str, default='saved_models/', help='the folder to save models')
    parse.add_argument('--log-dir', type=str, default='logs/', help='dir to save log information')
    parse.add_argument('--sim_dir', type=str, default='../Banana_Linux/Banana.x86_64', help='simulator directory')

    # dqn common parameters mainly used in train_dqn.py
    parse.add_argument('--n_episodes', type=int, default=int(1000), help='the total timesteps to train network')
    parse.add_argument('--max_t', type=int, default=int(300), help='maximum number of timesteps per episode')
    parse.add_argument('--eps_start', type=float, default=1, help='the initial exploration ratio')
    parse.add_argument('--eps_end', type=float, default=0.01, help='the final exploration ratio')
    parse.add_argument('--eps_decay', type=float, default=0.99, help='the exploration decay')
    parse.add_argument('--gamma', type=float, default=0.99, help='the discount factor of RL (eps_decay)')

    # training arguments
    parse.add_argument('--seed', type=int, default=0, help='the random seeds')
    parse.add_argument('--batch-size', type=int, default=32, help='the batch size of updating')
    parse.add_argument('--lr', type=float, default=1e-4, help='learning rate of the algorithm')
    parse.add_argument('--buffer-size', type=int, default=int(1e4), help='the size of the replay buffer')
    parse.add_argument('--tau', type=float, default=1e-2, help='for soft update of target parameters')
    parse.add_argument('--update_every', type=int, default=4, help='how often to update the network')

    # prioritized replay buffer
    parse.add_argument('--max_priority', type=float, default=1.0, help='the maximum priority for an experience')
    parse.add_argument('--alpha', type=float, default=0.6, help='the weighting factor a')
    parse.add_argument('--alpha_end', type=float, default=0.2, help='the weighting factor a')
    parse.add_argument('--beta', type=float, default=0.5, help='the weighting factor b')
    parse.add_argument('--beta_end', type=float, default=1.0, help='the weighting factor b')
    parse.add_argument('--ann_length', type=float, default=400, help='the weighting factor b')

    parse.add_argument('--eps', type=float, default=float(1e-3), help='the priority offset added to each priority')


    args = parse.parse_args()

    return args