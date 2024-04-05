import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gym
from torchvision.transforms import Resize
from PIL import Image

class ActorCriticPPO(nn.Module):
    def __init__(self, state_dim, action_dim, lr_actor=0.0001, lr_critic=0.001):
        super(ActorCriticPPO, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic

        self.actor = self.build_actor()
        self.critic = self.build_critic()

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)

    def build_actor(self):
        actor = nn.Sequential(
            nn.Linear(self.state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, self.action_dim),
            nn.Softmax(dim=-1)
        )
        return actor

    def build_critic(self):
        critic = nn.Sequential(
            nn.Linear(self.state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        return critic

    def get_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        action_probs = self.actor(state)
        action_dist = torch.distributions.Categorical(action_probs)
        action = action_dist.sample().item()
        return action

    def train(self, states, actions, advantages, discounted_rewards):
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        advantages = torch.FloatTensor(advantages)
        discounted_rewards = torch.FloatTensor(discounted_rewards)

        mu = self.actor(states)
        value = self.critic(states)
        advantage = advantages - value.squeeze()
        old_mu = mu.gather(1, actions.unsqueeze(1)).squeeze()

        # Actor loss
        ratio = torch.exp(self.log_prob(mu, actions) - self.log_prob(old_mu, actions))
        surr1 = ratio * advantage
        surr2 = torch.clamp(ratio, 1 - 0.2, 1 + 0.2) * advantage
        actor_loss = -torch.mean(torch.min(surr1, surr2))

        # Critic loss
        critic_loss = nn.MSELoss()(value.squeeze(), advantages)

        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()

        actor_loss.backward()
        critic_loss.backward()

        self.actor_optimizer.step()
        self.critic_optimizer.step()

    def log_prob(self, mu, action):
        action_dist = torch.distributions.Categorical(mu)
        return action_dist.log_prob(action)

    def discounted_rewards(self, rewards, gamma):
        discounted_rewards = np.zeros_like(rewards)
        running_add = 0
        for t in reversed(range(len(rewards))):
            running_add = running_add * gamma + rewards[t]
            discounted_rewards[t] = running_add
        return discounted_rewards

"""
Resizes image to 84x84
"""
def process_observation(observation):
    frame = Image.fromarray(observation)
    frame = frame.convert("L")
    frame = Resize((84, 84))(frame)
    frame = np.array(frame)
    return frame.flatten()

# Environment settings
env_name = "Breakout-v0"
env = gym.make(env_name)
state_dim = 84 * 84  # Preprocessed frame dimensions
action_dim = env.action_space.n

# Hyperparameters
lr_actor = 0.0001
lr_critic = 0.001
gamma = 0.99
epochs = 1000
batch_size = 64

# Initialize Actor-Critic PPO model
agent = ActorCriticPPO(state_dim, action_dim, lr_actor, lr_critic)

# Training loop
for epoch in range(epochs):
    states = []
    actions = []
    rewards = []
    dones = []
    values = []

    state, _ = env.reset()
    state = process_observation(state)
    while True:
        action = agent.get_action(state)
        next_state, reward, done, _, _ = env.step(action)
        next_state = process_observation(next_state)

        states.append(state)
        actions.append(action)
        rewards.append(reward)
        dones.append(done)
        values.append(agent.critic(torch.FloatTensor(state).unsqueeze(0)).detach().numpy()[0])

        state = next_state

        if done:
            discounted_rewards = agent.discounted_rewards(rewards, gamma)
            advantages = discounted_rewards - np.array(values)
            agent.train(states, actions, advantages, discounted_rewards)
            break

# Close environment
env.close()
