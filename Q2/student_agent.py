import gymnasium
import numpy as np
import torch
from train import PolicyNetwork

# Do not modify the input of the 'act' function and the '__init__' function. 
class Agent(object):
    """Agent that acts randomly."""
    def __init__(self):
        self.action_space = gymnasium.spaces.Box(-1.0, 1.0, (1,), np.float64)
        self.state_dim = 5
        self.action_dim = 1
        self.lr_pi = 0.001
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.network = PolicyNetwork(self.state_dim, self.action_dim, self.lr_pi).to(self.device)
        self.network.load_state_dict(torch.load('sac_actor_final.pt'))

    def act(self, observation):
        with torch.no_grad():
            action, _ = self.network.sample(torch.FloatTensor(observation).to(self.device))
            return action
        
        return self.action_space.sample()
