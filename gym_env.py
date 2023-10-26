import numpy as np
import gym
from gym import error, spaces, utils
from gym.utils import seeding

from Cartpole.model import Cartpole
from Pendulum.model import Pendulum
from Quadrotor.model import Quadrotor


meta_dict = {
    "Cartpole": {
        "model": Cartpole,
        "Q": np.diag([60, 1.5, 180, 45]),
        "R": np.diag([1.0]),
    },
    "Pendulum": {
        "model": Pendulum,
        "Q": np.diag([1.0, 1.0]),
        "R": np.diag([2.0]),
    },
    "Quadrotor": {
        "model": Quadrotor,
        "Q": np.diag([15,15,15,5,5,5,0,0,0,0.5,0.5,0.03])*0.8,
        "R": np.diag([1,1,1,1])*(0.5e-7)*9,
    }
}

class GymEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self,env_name):
        assert env_name in meta_dict.keys()
        super(GymEnv, self).__init__()

        self.env_name = env_name
        self.env_cls = meta_dict[env_name]["model"]
        self.env = self.env_cls()
        self.Q = meta_dict[env_name]["Q"]
        self.R = meta_dict[env_name]["R"]

        self.action_space = spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(self.env.dim_m,),
                dtype=np.float32
            )
        
        if self.env_name in ["Cartpole","Pendulum"]:
            self.observation_space = spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(self.env.dim_n,),
                    dtype=np.float32
                )
        else:
            self.observation_space = spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(self.env.measure_dim,),
                dtype=np.float32
            )
        
        if env_name == "Cartpole":
            self.done = self.done_Cartpole
        elif env_name == "Pendulum":
            self.done = self.done_Pendulum
        else:
            self.done = self.done_Quadrotor
        
        if env_name == "Cartpole":
            self.dt = 0.01
        else:
            self.dt = 0.02

        self.seed()

    def step(self, action):
        self.env.integrate(action,[0,self.dt])
        state = self.env.Read_sensor()
        reward = -(state.T @ self.Q @ state + action @ self.R @ action).item()
        done = self.done(state)
        return state,reward,done
    
    def done_Cartpole(self, state):
        return (state < -10).any() or (state > 10).any()
        # return False

    def done_Pendulum(self, state):
        return not self.env.In_domain(state)
        
    def done_Quadrotor(self,state):
        return (state < -10).any() or (state > 10).any()
    
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        return self.env.randomly_initialize()

    def render(self, mode='human', close=False):
        """
        This methods provides the option to render the environment's behavior to a window 
        which should be readable to the human eye if mode is set to 'human'.
        """
        pass

    def close(self):
        """
        This method provides the user with the option to perform any necessary cleanup.
        """
        pass
