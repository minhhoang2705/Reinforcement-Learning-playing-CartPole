import numpy as np
import random 
import torch
import torch.nn as nn
import torch.optim as optim

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = []
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.learning_rate = 0.001
        self.model = self.build_model()
    
    def _build_model(self):
        model = nn.Sequential(
            nn.Linear(self.state_size, 24),
            nn.ReLU(),
            nn.Linear(24, 24),
            nn.ReLU(),
            nn.Linear(24, self.action_size)
        )
        return model
    
    def act(self, state):
        """
        Choose action based on epsilon-greedy policy
        """
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state = torch.tensor(state, dtype=torch.float)
        act_values = self.model(state)
        return torch.argmax(act_values).item()
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * torch.max(self.model(next_state))
            target_f = self.model(torch.FloatTensor(state))
            target_f[action] = target
            self.model.zero_grad()
            loss = nn.MSELoss()(target_f, self.model(torch.FloatTensor(state)))
            loss.backward()
            for param in self.model.parameters():
                param.data -= self.learning_rate * param.grad.data
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            