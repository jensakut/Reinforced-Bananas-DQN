[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135619-d90f2f28-7d12-11e8-8823-82b970a54d7e.gif "Trained Agent"
### Udacity Deep Reinforcement Learning Nanodegree
## Project 1: Navigation
# Collecting Bananas with DQN

## Introduction

The project teaches a DQN in a banana-collecting environment. The agent navigates in an environment to collect yellow
bananas while avoiding black ones. 
A trained agent may act like this example: 

![Trained Agent][image1]

A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana.  Thus, the goal of your agent is to collect as many yellow bananas as possible while avoiding blue bananas.  

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around agent's forward direction.  Given this information, the agent has to learn how to best select actions.  Four discrete actions are available, corresponding to:
- **`0`** - move forward.
- **`1`** - move backward.
- **`2`** - turn left.
- **`3`** - turn right.

The task is episodic, and in order to solve the environment, the agent must get an average score of +13 over 100 consecutive episodes.

### Getting Started

1. Download the environment from one of the links below.  You need only select the environment that matches your operating system:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)
    
    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

    (_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux_NoVis.zip) to obtain the environment.

For the visual environment the downloads are as follows: 
You need only select the environment that matches your operating system:
- Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/VisualBanana_Linux.zip)
- Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/VisualBanana.app.zip)
- Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/VisualBanana_Windows_x86.zip)
- Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/VisualBanana_Windows_x86_64.zip)


2. Extract the files into the root folder. If used in non-linux 64 bit environment, change the path in the navigation*.py files in the following line:

        env = UnityEnvironment(file_name="Banana_Linux/Banana.x86_64")

3a. Use Anaconda to install the 'pytorch' environment 

        conda env create -f environment.yml
        conda activate pytorch 
        
3b. alternatively, use the requirements.txt to install via pip in your favorite python distribution

        pip install -r requirements.txt (Python 2), 
        pip3 install -r requirements.txt (Python 3)


4. Train the agent with the arguments in the file dqn/arguments.py by executing in folder dqn

        python navigation_dqn.py  



## Background: 

Instead of storing q-values in tables, deep q learning uses a deep neural network to approximate the value function. 
To mitigate the instability of representing value functions using neural networks, two key features are essential: 
- Experience Replay
- Fixed Q targets. 


    Riedmiller, Martin. "Neural fitted Q iteration–first experiences with a data efficient neural reinforcement learning method." European Conference on Machine Learning. Springer, Berlin, Heidelberg, 2005.
    http://ml.informatik.uni-freiburg.de/former/_media/publications/rieecml05.pdf

    Mnih, Volodymyr, et al. "Human-level control through deep reinforcement learning." Nature518.7540 (2015): 529.
    http://www.davidqiu.com:8888/research/nature14236.pdf


### Experience Replay

A simple algorithm may collect a trajectory, learn from it, and collect a new one. But, a single trajectory does not necessarily
represent the optimal value function of a given problem. Thus, the replay buffer stores any collected experience tuple 
state, action, reward, next state. 
A sampled batch from this experience buffer is called *experience replay*. Learning from this helps learning a more representative and generalized 
function. Individual tuples are revisited multiple times, and rare occurences can be recalled. 
The experience generally helps to improve data efficiency. 

### Fixed Q Targets 

Fixed q target is an essential method to improve stability for deep q learning introduced by deepmind. 
Q Learning updates a guess with a guess, and this can lead to harmful oscillations during training. By learning the weights, 
the target moves, thus the gradient used for learning increases. Thus, the gradient doesn't decrease with learning. Hence the oscillations. 

The idea of fixed q targets is as follows. 
- Using a target network with fixed parameters w- for estimating the TD target
- At every Tau step, copy the parameters from the local network to update the target network

![Bellman equation fixed q targets](assets/bellman_fixed_q_targets.png?raw=true "Bellman equation for fixed q targets")

The implementation is straight forward. 
- Create a local and a target network 
- Create a function that copies the weights of the local network into the target network
- During training, calculate the td target using both the target network. Update the local network using this target. 
Update the target network every tau steps by copying the weights of the local network. 


### Further improvements of the dqn algorithm are 
- [Double DQN](https://arxiv.org/abs/1509.06461)
- [Dueling DQN (DDQN)](https://arxiv.org/abs/1511.06581)
- [Prioritized experience replay](https://arxiv.org/abs/1511.05952)
- [Learning from multi-step bootstrap targets](https://arxiv.org/abs/1602.01783)
- [Distributional DQN](https://arxiv.org/abs/1707.06887)
- [Noisy DQN](https://arxiv.org/abs/1706.10295)

The results are in [summarized in rainbow dqn paper by Hessel et. al.](https://arxiv.org/abs/1710.02298)



# Report 

The repository contains the following algorithms with optimized parameters : 
- Vanilla DQN 
- Double DQN 
- Dueling DQN
- Prioritized experience replay 

Interestingly, both the Dueling as well as Prioritized Experience Replay did not improve on the baseline. Though 
they both are able to solve the environment with a score > 13, they neither improve the high score with additional 
training, nor do they make training faster. 
For manual optimization, the number of parameters is the main challenge to manage. Therefore, images with all 
training results are saved in the folder results. 
The takeaway is summarized in the following chapters. 

## Model 
The Double DQN uses a simple network architecture exhibiting fully
 connected layers. This simple architecture is suitable to learn the required 
 Q_values without a lot of data. 
 The dqn_model_test loads the dqn network and prints the nodes. 

  (fc1): Linear(in_features=37, out_features=64, bias=True)
  (fc2): Linear(in_features=64, out_features=64, bias=True)
  (fc3): Linear(in_features=64, out_features=4, bias=True)
  (state_value): Linear(in_features=64, out_features=1, bias=True)

The activation functions are relu and as Adam Optimizer was chosen. 

## Parameters

The following parameters help the network to learn fast. It is interesting to note 
that the Epsilon decays faster and the learning rate is higher to boost fast learning of a 
relatively simple task. 

The parameters from the Atari Papers work well. With the small network, a score of 13 is achieved in 550 episodes. 
The parameters from the prioritized experience replay are: 
Learning rate 1e-4, learn every 4 steps from 32 episodes. If the agent learns every 16 steps from 256 episodes, the 
steps are fewer, but the episodes are faster due to the usage of a graphics card with enough memory.  
The average score is improved from 14.3 to 15 in fewer episodes.

A faster training can be achieved using the following parameters: 
### Experience Buffer 10000
Since the learning is quick, the experience buffer is smaller, because older examples 
do not represent the quickly adapted policy that well. 
 
### n_episodes 700
The network overfits beyond 700, indicated by reduced scores. The useful number of episodes varies greatly among the parameters. 
Basically if the network does not improve or substantially (more than 2 points from max) decreases, training is futile. 
  
- max_t=300 
The environment does exactly 300 steps independently of the agent's actions. 
- eps_start=1.0
- eps_end=0.01
- eps_decay=0.99
At start, mainly random exploration helps to formulate a rough concept of what is going on. If epsilon is not decayed 
at that rate, the network does not achieve a high score. For a non-small network 
- BATCH_SIZE = 256  # minibatch size stabilizes the learning parameters
- GAMMA = 0.99  # discount factor
- TAU = 1e-2  # for soft update of target parameters
- LR = 1e-4  # learning rate. 
- UPDATE_EVERY = 16  # how often to update the network
- BUFFER_SIZE = int(1e4)  # replay buffer size

The environment is solved in ~350 episodes with an average score of 100 episodes above 13. After 560 Episodes, 
a score of 15 is achieved. 



## Troubleshooting 

        unityagents.exception.UnityEnvironmentException: Couldn't launch the Banana environment. Provided filename does not match any environments.

means the unityenvironment is not found. The path is configured in dqn/navigation_dqn.py in line 68. Using linux, the 
downloaded folder should be placed in this root directory. 