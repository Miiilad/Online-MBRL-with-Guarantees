from datetime import datetime

from ADP import ADP
from SGA import SGA
from SOL_controller import SOL_controller
from env_hparams import env_hparams
from gym_env import GymEnv
from PPO.PPO import PPO

env_name = "Pendulum" # env_name must be one of ["Cartpole", "Pendulum", "Quadrotor"]
random_seed = "SGA" # random_seed must be one of [1,2,3,4,5,6,"SOL","SGA","ADP"]

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

if random_seed in [1,2,3,4,5,6]:
    directory = "PPO_preTrained" + '/' + env_name + '/'
    checkpoint_path = directory + "PPO_{}_{}_{}.pth".format(env_name, random_seed, run_num_pretrained)
    print("loading network from : " + checkpoint_path)

    ppo_agent.load(checkpoint_path)


################################### Testing ###################################

total_test_episodes = 100
test_running_reward = 0
total_time = 0
speed = 0


if random_seed == "SOL":
    control = sol_controller.select_control
elif random_seed == "SGA":
    control = sga_controller.select_control
elif random_seed == "ADP":
    control = adp_controller.select_control
else:
    control = ppo_agent.select_action

for ep in range(1, total_test_episodes+1):
    ep_reward = 0
    state = env.reset()
    
    start_time = datetime.now()
    
    for t in range(1, max_ep_len+1):
        action = control(state)
        state, reward, done = env.step(action)
        ep_reward += reward
        
        if done:
            ep_reward += reward * (max_ep_len-t)
            break

    end_time = datetime.now()
    ep_reward = -ep_reward / max_ep_len

    speed += (end_time - start_time).total_seconds()
    total_time += t
    
    print('Episode: {} \t\t Speed: {} \t\t t: {}'.format(ep, (end_time - start_time)/t, t))
    # clear buffer    
    ppo_agent.buffer.clear()

    test_running_reward +=  ep_reward
    print('Episode: {} \t\t Avg Control Cost: {}'.format(ep, round(ep_reward, 5)))
    ep_reward = 0
env.close()


print("============================================================================================")

avg_test_reward = test_running_reward / total_test_episodes
avg_test_reward = round(avg_test_reward, 5)
print("average test reward : " + str(avg_test_reward))
print("average speed : " + str(round(speed / total_time, 5)))

print("============================================================================================")

