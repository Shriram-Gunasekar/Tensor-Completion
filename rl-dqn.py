import torch
import torch.nn as nn
import numpy as np
import random
from collections import deque
import gym

# Define the DQN model for tensor completion
class DQN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.fc2(x)
        return x

# Define the RL agent
class Agent:
    def __init__(self, input_dim, output_dim, hidden_dim, lr=0.001, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.memory = deque(maxlen=2000)
        self.model = DQN(input_dim, hidden_dim, output_dim)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.output_dim)
        q_values = self.model(torch.tensor(state).float())
        return torch.argmax(q_values).item()

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        batch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in batch:
            target = reward
            if not done:
                target = reward + self.gamma * torch.max(self.model(torch.tensor(next_state).float()))
            q_values = self.model(torch.tensor(state).float())
            target_f = q_values.clone().detach()
            target_f[action] = target
            self.optimizer.zero_grad()
            loss = self.criterion(q_values, target_f)
            loss.backward()
            self.optimizer.step()
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# Define a simple environment for tensor completion
class TensorCompletionEnv(gym.Env):
    def __init__(self, tensor_size):
        self.tensor_size = tensor_size
        self.tensor = np.random.rand(*self.tensor_size)
        self.missing_mask = np.random.choice([True, False], size=self.tensor_size, p=[0.2, 0.8])
        self.state = self.tensor.copy()
        self.state[self.missing_mask] = np.nan
        self.action_space = gym.spaces.Discrete(10)  # Example: 10 possible actions (filling missing values)
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=self.tensor_size, dtype=np.float32)

    def reset(self):
        self.state = self.tensor.copy()
        self.state[self.missing_mask] = np.nan
        return self.state

    def step(self, action):
        reward = 0
        done = False
        # Example: Fill missing values based on the action
        self.state[self.missing_mask] = action / 10.0
        # Example: Calculate reward based on reconstruction error
        reward -= np.sum((self.state - self.tensor) ** 2)
        return self.state, reward, done, {}

# Example usage
tensor_size = (10, 10)
env = TensorCompletionEnv(tensor_size)
agent = Agent(np.prod(tensor_size), env.action_space.n, hidden_dim=64)

num_episodes = 1000
batch_size = 32

for episode in range(num_episodes):
    state = env.reset()
    total_reward = 0
    done = False
    while not done:
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        agent.remember(state.flatten(), action, reward, next_state.flatten(), done)
        state = next_state
        total_reward += reward
    agent.replay(batch_size)
    print(f'Episode {episode+1}/{num_episodes}, Total Reward: {total_reward}')

# Evaluate the agent on a new tensor
new_tensor = np.random.rand(*tensor_size)
new_missing_mask = np.random.choice([True, False], size=tensor_size, p=[0.2, 0.8])
new_state = new_tensor.copy()
new_state[new_missing_mask] = np.nan
action = agent.act(new_state.flatten())
new_state[new_missing_mask] = action / 10.0
