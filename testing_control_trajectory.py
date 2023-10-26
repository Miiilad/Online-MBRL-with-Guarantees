from datetime import datetime
from matplotlib import pyplot as plt
import seaborn as sns

import numpy as np
import torch

from ADP import ADP
from SGA import SGA
from SOL_controller import SOL_controller
from env_hparams import env_hparams
from gym_env import GymEnv
from PPO.PPO import PPO

env_name = "Cartpole" # env_name must be one of ["Cartpole", "Pendulum", "Quadrotor"]
random_seed = 1 # random_seed must be one of [1,2,3,4,5,6]

env = GymEnv(env_name)

####### initialize environment hyperparameters ######

has_continuous_action_space = env_hparams[env_name]["has_continuous_action_space"]

max_ep_len = env_hparams[env_name]["max_ep_len"]
max_training_timesteps = env_hparams[env_name]["max_training_timesteps"]

print_freq = env_hparams[env_name]["print_freq"]
log_freq = env_hparams[env_name]["log_freq"]
save_model_freq = env_hparams[env_name]["save_model_freq"]

action_std = env_hparams[env_name]["action_std"]
action_std_decay_rate = env_hparams[env_name]["action_std_decay_rate"]
min_action_std = env_hparams[env_name]["min_action_std"]
action_std_decay_freq = env_hparams[env_name]["action_std_decay_freq"]

################################### Initialize SOL controller ###################################

sol_controller = SOL_controller(env_name)

################################### Initialize SGA controller ###################################

sga_controller = SGA(env_name)

################################### Initialize ADP controller ###################################

adp_controller = ADP(env_name)

################################### Initialize PPO agent ###################################
has_continuous_action_space = env_hparams[env_name]["has_continuous_action_space"]
action_std = env_hparams[env_name]["min_action_std"]
max_ep_len = env_hparams[env_name]["max_ep_len"]

# state space dimension
state_dim = env.observation_space.shape[0]

# action space dimension
if has_continuous_action_space:
    action_dim = env.action_space.shape[0]
else:
    action_dim = env.action_space.n

K_epochs = 80
eps_clip = 0.2
gamma = 0.99

lr_actor = 0.0003
lr_critic = 0.001

run_num_pretrained = 0
ppo_agent = PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space, action_std)

directory = "PPO_preTrained" + '/' + env_name + '/'
checkpoint_path = directory + "PPO_{}_{}_{}.pth".format(env_name, random_seed, run_num_pretrained)
print("loading network from : " + checkpoint_path)

ppo_agent.load(checkpoint_path)

################################### Testing ###################################

def run_episode(algorithm):
    rewards = np.zeros((max_ep_len))

    state = env.reset()

    if algorithm == "SOL":
        control = sol_controller.select_control
    elif algorithm == "SGA":
        control = sga_controller.select_control
    elif algorithm == "ADP":
        control = adp_controller.select_control
    else:
        control = ppo_agent.select_action

    for t in range(1, max_ep_len+1):
        action = control(state)
        state, reward, done = env.step(action)
        rewards[t-1] = -reward
        
        if done:
            rewards[t-1:] = -reward
            break

    env.close()

    return rewards

num_init_cond = 100

# smooth out rewards to get a smooth and a less smooth (var) plot lines
window_len_smooth = 20
min_window_len_smooth = 1
linewidth_smooth = 1.5
alpha_smooth = 1

window_len_var = 1
min_window_len_var = 1
linewidth_var = 2
alpha_var = 0.1


ax = plt.gca()
sns.set_theme()

if env_name == "Pendulum":
    algorithm_list = ["PPO","SOL","SGA","ADP"]
elif env_name == "Cartpole":
    # algorithm_list = ["PPO","SOL","ADP"]
    algorithm_list = ["SOL","ADP"]
else:
    algorithm_list = ["PPO","SOL"]


for algorithm in algorithm_list:
    print("Running trajectories for " + algorithm)
    env.seed(1)
    rewards = np.zeros((num_init_cond,max_ep_len))
    for i in range(num_init_cond):
        rewards[i,:] = run_episode(algorithm)
    rewards_mean = np.mean(rewards,axis=0)
    rewards_var = np.std(rewards,axis=0)
    # rewards_mean_smooth = np.convolve(rewards_mean,np.ones((window_len_smooth))/window_len_smooth, mode="same")
    # rewards_mean_var = np.convolve(rewards_mean,np.ones((window_len_var))/window_len_var, mode="same")

    timestep = np.linspace(0.02,30,max_ep_len)

    colors = {"PPO":"r", "SOL": "g", "SGA": "b", "ADP": "orange"}

    ax.plot(timestep[:-window_len_smooth], rewards_mean[:-window_len_smooth], color=colors[algorithm], label=algorithm)
    
    ax.fill_between(timestep[:-window_len_smooth], 
                    (rewards_mean-rewards_var)[:-window_len_smooth], 
                    (rewards_mean+rewards_var)[:-window_len_smooth], 
                    alpha=alpha_var,
                    color=colors[algorithm])
    
ax.set(ylabel="Control Cost")
ax.set(xlabel="Time (s)")
ax.set(title=env_name)

# keep only reward_smooth in the legend and rename it
handles, labels = ax.get_legend_handles_labels()

ax.legend(handles,algorithm_list)
if env_name == "Pendulum":
    ax.set_ylim([-0.1,0.6])
elif env_name == "Cartpole":
    ax.set_ylim([-0.1,1])

plt.tight_layout()
save_dir = 'fig_trajectory_' + env_name + '.pdf'
plt.savefig(save_dir,format='pdf')
print("Saved at: " + save_dir)
