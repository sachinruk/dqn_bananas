import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        "*** YOUR CODE HERE ***"
        h = 20
        self.model = nn.Sequential(nn.Linear(state_size, h),
                      nn.ReLU(),
                      nn.Linear(h, h),
                      nn.ReLU(),
                      nn.Linear(h, action_size)
                      )
        self.criterion = nn.MSELoss()
#         self.optimizer = optim.Adam(model.parameters())

    def forward(self, state):
        """Build a network that maps state -> action values."""
        return self.model.forward(state)
    
class QNetworkCNN(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(QNetworkCNN, self).__init__()
        self.seed = torch.manual_seed(seed)
        "*** YOUR CODE HERE ***"
        h = 20
        self.conv1 = nn.Conv2d(state_size[-1], 4, 3, padding=1)
        self.conv2 = nn.Conv2d(4, 8, 3, padding=1)
        self.conv3 = nn.Conv2d(8, 16, 3, padding=1)
        self.conv4 = nn.Conv2d(16, 16, 3, padding=1)
        self.conv5 = nn.Conv2d(16, 16, 3, padding=1)
        
        self.pool = nn.MaxPool2d(2, ceil_mode=True)
        
        self.fc1 = nn.Linear(3*3*16,10)
        self.fc2 = nn.Linear(10, action_size)
        
        self.criterion = nn.MSELoss()
#         self.optimizer = optim.Adam(model.parameters())

    def forward(self, x):
        """Build a network that maps state -> action values."""
#         from IPython.core.debugger import Tracer; Tracer()()
        x = self.pool(F.relu(self.conv1(x)))
        
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        x = self.pool(F.relu(self.conv5(x)))
        x = x.reshape(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    

class QNetworkDuellingCNN(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(QNetworkDuellingCNN, self).__init__()
        self.seed = torch.manual_seed(seed)
        "*** YOUR CODE HERE ***"
        h = 20
        self.conv1 = nn.Conv2d(state_size[-1], 4, 3, padding=1)
        self.conv2 = nn.Conv2d(4, 8, 3, padding=1)
        self.conv3 = nn.Conv2d(8, 16, 3, padding=1)
        self.conv4 = nn.Conv2d(16, 16, 3, padding=1)
        self.conv5 = nn.Conv2d(16, 16, 3, padding=1)
        
        self.pool = nn.MaxPool2d(2, ceil_mode=True)
        
        self.fcval = nn.Linear(3*3*16,10)
        self.fcval2 = nn.Linear(10, 1)
        self.fcadv = nn.Linear(3*3*16,10)
        self.fcadv2 = nn.Linear(10, action_size)
        
        
        self.criterion = nn.MSELoss()
#         self.optimizer = optim.Adam(model.parameters())

    def forward(self, x):
        """Build a network that maps state -> action values."""
#         
        x = self.pool(F.relu(self.conv1(x)))
        
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        x = self.pool(F.relu(self.conv5(x)))
        x = x.reshape(x.shape[0], -1)

        
        advantage = F.relu(self.fcadv(x))
        advantage = self.fcadv2(advantage)
        advantage = advantage - torch.mean(advantage, dim=-1, keepdim=True)
        
        value = F.relu(self.fcval(x))
        value = self.fcval2(value)
#         from IPython.core.debugger import Tracer; Tracer()()
        return value + advantage