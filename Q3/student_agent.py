import gymnasium as gym
import numpy as np
import torch
from train import PolicyNetwork

# Do not modify the input of the 'act' function and the '__init__' function. 
class Agent(object):
    """Agent that acts randomly."""
    def __init__(self):
        self.action_space = gym.spaces.Box(-1.0, 1.0, (21,), np.float64)
        self.state_dim = 67
        self.action_dim = 21
        self.lr_pi = 0.001
        self.device = torch.device("cpu")
        self.network = PolicyNetwork(self.state_dim, self.action_dim, self.lr_pi).to(self.device)
        self.network.load_state_dict(torch.load('sac_actor_final.pt', map_location=torch.device('cpu')))

    def act(self, observation):
        with torch.no_grad():
            action, _ = self.network.sample(torch.FloatTensor(observation).to(self.device))
            return action
        
        return self.action_space.sample()
