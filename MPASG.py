import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import time
from collections import deque
from pulp import LpProblem, LpMaximize, LpVariable, lpSum, value
import pulp
import matplotlib.pyplot as plt
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return np.array(state), action, reward, np.array(next_state), done

    def __len__(self):
        return len(self.buffer)

class DQNAgent:
    def __init__(self, state_dim,  buffer_capacity=20000, batch_size=32, gamma=0.95, lr=0.001):
        self.gamma = gamma
        self.batch_size = batch_size
        self.q_network = DQN(state_dim, state_dim).to(device)
        self.target_network = DQN(state_dim, state_dim).to(device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.replay_buffer = ReplayBuffer(buffer_capacity)
        self.loss_fn = nn.SmoothL1Loss().to(device)


    def train(self):
        if len(self.replay_buffer) < self.batch_size:
            return
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        states = torch.FloatTensor(states).to(device)
        actions = torch.LongTensor(actions).to(device)
        rewards = torch.FloatTensor(rewards).to(device)
        next_states = torch.FloatTensor(next_states).to(device)
        dones = torch.FloatTensor(dones).to(device)
        q_values = (self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1))
        next_actions = self.q_network(next_states).argmax(dim=1, keepdim=True)
        next_q_values = self.target_network(next_states).gather(1, next_actions).squeeze(1)
        expected_q_values = rewards + self.gamma * next_q_values * (1 - dones)
        loss = self.loss_fn(q_values, expected_q_values.detach())
        lossValue = loss.item()
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.q_network.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        return lossValue

    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

def train_dqn(num_rules, agent, num_episodes, num_step, rule_scores, epsilon_start=1.0, epsilon_end=0.01):
    epsilon = epsilon_start
    epsilon_decay_rate = (epsilon_start - epsilon_end) / (num_episodes)
    rewards = []
    lossValues = []
    process = 0
    for episode in range(num_episodes):
        S = []
        state = [0] * num_rules
        selected_actions = list(range(num_rules))
        lossValue = 0
        for t in range(num_step):
            action = agent.select_action(state, epsilon, selected_actions)
            if action in selected_actions:
                selected_actions.remove(action)
            S.append(action)
            next_state = state.copy()
            next_state[action] = 1
            reward, done = agent.get_reward(S, rule_scores)
            agent.replay_buffer.push(state, action, reward, next_state, done)
            lossValue = agent.train()
            state = next_state.copy()
        epsilon = max(epsilon_end, epsilon - epsilon_decay_rate)
        if (episode+1)%(50) ==0:
           agent.update_target_network()
        if (episode + 1) % (num_episodes / 10) == 0:
            process += 10
            print("进度:", process, "%")
        rewards.append(reward * 100)
        if lossValue is not None:
            lossValues.append(lossValue)
    return rewards, lossValues, S

start_time = time.perf_counter()
loaded_rules_scores = np.loadtxt('rules_scores2.txt', dtype=int)
num_rules = 200
state_dim = num_rules
agent = DQNAgent(num_rules)
num_episodes = 20000
num_step = 20
S1 = []
visited_states = set()
Sbuffer = deque(maxlen=50)
Sbuffer = [sum(rule_scores)/400]
rewards, lossValues,S = train_dqn(num_rules, agent, num_episodes, num_step, rule_scores)
end_time = time.perf_counter()
execution_time = end_time - start_time
with open("rewardMPASG.txt", "w") as file:
    file.write(f"{rewards}\n")
print(f"Code took {execution_time} seconds")
selected_rules = [rules_matrix[i] for i in S]
rule_matrix1 = agent.rule_matrix(selected_rules)
count_rule,tran, fail_penalty = agent.programming(rule_matrix1, rule_scores, S)
if fail_penalty:
    reward='分组失败'
else:
    reward = agent.calcScore(scores_matrix, count_rule, S)
    for i in range(len(S)):
        print('采用规则为:',S[i],'使用次数:',count_rule[i])
    print('小组得分为',reward*100)
fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(10, 4))
generation = range(0, len(rewards))
window_size = int(num_episodes/50)
y_smoothed = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
x_smoothed = generation[window_size-1:]
axs[0].scatter(generation, rewards, label='Generations')
axs[0].plot(x_smoothed, y_smoothed, color='red')
axs[0].set_title('Rewards Over Generations')
axs[0].set_xlabel('Generations')
axs[0].set_ylabel('Rewards')
axs[0].grid(True)
axs[0].legend()

axs[1].plot(np.arange(0, len(lossValues)), lossValues, label='Training Loss')
axs[1].set_title('Loss Function Over Epochs')
axs[1].set_xlabel('Epochs')
axs[1].set_ylabel('Loss')
axs[1].legend()
plt.show()