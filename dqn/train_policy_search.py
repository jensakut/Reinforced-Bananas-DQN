from dqn.Agent import Agent
from dqn.PolicySearch import PolicySearch
from dqn.UnityWrapper import UnityWrapper
from dqn.arguments import get_args
from dqn.train import dqn

# main function
if __name__ == "__main__":

    # get arguments
    args = get_args()

    env_wrapper = UnityWrapper(args)
    state_size, action_size, _ = env_wrapper.get_env_info()
    policy_search = PolicySearch(args)

    # run agent once to get rating for initial configuration
    agent = Agent(args,
                  state_size=state_size,
                  action_size=action_size,
                  filename='')
    scores, scores_mean = dqn(agent, args, env_wrapper)

    for i in range(100):
        args = policy_search.generate_policy(args, rewards=[scores, scores_mean])

        agent = Agent(args,
                      state_size=state_size,
                      action_size=action_size,
                      filename='')
        scores, scores_mean = dqn(agent, args, env_wrapper)
