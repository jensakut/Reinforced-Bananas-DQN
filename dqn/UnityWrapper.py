from unityagents import UnityEnvironment


class UnityWrapper():

    def __init__(self, args):
        # initialize the environment
        self.env = UnityEnvironment(file_name=args.sim_dir)

        # for reset
        self.train = args.train
        # get the default brain
        self.brain_name = self.env.brain_names[0]
        self.brain = self.env.brains[self.brain_name]

        # reset the environment
        self.env_info = self.env.reset(train_mode=args.train)[self.brain_name]

        # number of agents in the environment
        print('Number of agents:', len(self.env_info.agents))

        # number of actions
        action_size = self.brain.vector_action_space_size
        print('Number of actions:', action_size)

        # examine the state space
        state = self.env_info.vector_observations[0]
        print('States look like:', state)
        state_size = len(state)
        print('States have length:', state_size)

    def get_env_info(self):
        # number of agents in the environment
        print('Number of agents:', len(self.env_info.agents))
        # number of actions
        action_size = self.brain.vector_action_space_size
        print('Number of actions:', action_size)
        # examine the state space
        state = self.env_info.vector_observations[0]
        print('States look like:', state)
        state_size = len(state)
        print('States have length:', state_size)
        return state_size, action_size, state

    def reset(self, brain_name=None, brain_id=0):
        if brain_name is None:
            brain_name = self.brain_name
        env_info = self.env.reset(train_mode=self.train)[brain_name]
        next_state = env_info.vector_observations[brain_id]  # get the next state
        return next_state

    def step(self, action, brain_name=None, brain_id=0):
        if brain_name is None:
            brain_name = self.brain_name
        env_info = self.env.step(action)[brain_name]
        next_state = env_info.vector_observations[brain_id]  # get the next state
        reward = env_info.rewards[brain_id]  # get the reward
        done = env_info.local_done[brain_id]  # see if episode has finished
        return reward, next_state, done
