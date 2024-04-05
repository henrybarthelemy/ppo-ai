import numpy as np
import gym
from networks import FeedForwardNN, FeedForwardImageNetwork
import torch
from torch.distributions import MultivariateNormal, Categorical
from torch.optim import Adam
import torch.nn as nn
from PIL import Image

class PPO:
    def __init__(self, env):
        self.device = 'cpu'
        # initialize all the hyper parameters in spot to help with readability
        self._init_hyperparameters()
        # environment we are working with
        self.env = env
        # observation and action space
        self.obs_dim = env.observation_space.shape[0]
        self.act_dim = env.action_space.n
        # actor and critic network
        self.actor = FeedForwardImageNetwork(self.act_dim).to(self.device)
        self.critic = FeedForwardImageNetwork(1).to(self.device)
        # Matrix for action exploration calculations
        self.cov_var = torch.full(size=(self.act_dim,), fill_value=0.5)
        self.cov_mat = torch.diag(self.cov_var)
        # Optimizer
        self.actor_optim = Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optim = Adam(self.critic.parameters(), lr=self.lr)



    """
    Initializes all the hyperparameters for a PPO training
    """
    def _init_hyperparameters(self):
        self.timesteps_per_batch = 4800 # timesteps per batch
        self.max_timesteps_per_episode = 1600 # timesteps per episode
        self.gamma = 0.95 # discount factor
        self.n_updates_per_iter = 5 # number of updates per iteration
        self.clip = 0.2 # clip hyperparam recommended by open ai paper
        self.lr = 0.005

    def learn(self, total_timesteps):
        t_so_far = 0
        while t_so_far < total_timesteps:
            batch_obs, batch_acts, batch_rews, batch_log_probs, batch_rtgs, batch_lens = self.rollout()
            # get value of each obs
            V, _ = self.evaluate(batch_obs, batch_acts)
            # Calculate advantage
            A_k = batch_rtgs - V.detach()
            # Normalize advantages
            A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10)  # small term to avoid division by zero
            for _ in range(self.n_updates_per_iter):
                # calc V_phi and pi_theta(a_t | s_t)
                V, curr_log_probs = self.evaluate(batch_obs, batch_acts)
                # Calculate ratios
                ratios = torch.exp(curr_log_probs - batch_log_probs)
                # surrogate losses (we will find min of this for loss)
                surr1 = ratios * A_k
                surr2 = torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * A_k
                actor_loss = (-torch.min(surr1, surr2)).mean()
                # Calculates backpropagation for actor neural network
                self.actor_optim.zero_grad()
                # retain graphs says go first to avoid buffer issues with critic
                actor_loss.backward(retain_graph=True)
                self.actor_optim.step()
                # Do same for critic
                critic_loss = nn.MSELoss()(V, batch_rtgs)
                self.critic_optim.zero_grad()
                critic_loss.backward()
                self.critic_optim.step()
            t_so_far += np.sum(batch_lens)




    """
    Use our critic to evaluate the value of each observation in a batch
    """
    def evaluate(self, batch_obs, batch_acts):

        Vs = []
        log_probs = []
        with torch.no_grad():
            for j in range(len(batch_obs)):
                V = self.critic(batch_obs[j]).squeeze()
                pi = self.actor.pi(batch_obs[j], softmax_dim=0)
                m = Categorical(pi)
                a = m.sample().item()
                pi_a = pi[0][a].item()
                Vs.append(a)
                log_probs.append(pi_a)
        return Vs, log_probs

    """
    Gets an action and log probability for it given the current policy
    """
    def get_action(self, obs):
        if isinstance(obs, np.ndarray):
            obs = torch.from_numpy(obs).float().to(self.device)
        with torch.no_grad():
            pi = self.actor.pi(obs, softmax_dim=0)
            m = Categorical(pi)
            a = m.sample().item()
            pi_a = pi[0][a].item()
            return a, pi_a


    """
    Collect a batch of data
    """
    def rollout(self):
        # batch data
        batch_obs = []  # observation
        batch_acts = [] # actions
        batch_log_probs = [] # log probabilities
        batch_rews = [] # rewards
        batch_lens = [] # length of episodes

        t = 0 # cur training step

        while t < self.timesteps_per_batch:
            # ep rewards
            ep_rews = []

            obs, _ = self.env.reset()
            obs = self.process_observation(obs)
            for ep_t in range(self.max_timesteps_per_episode):
                t += 1
                batch_obs.append(obs)
                action, log_prob = self.get_action(obs)
                obs, rew, done, _, _ = self.env.step(action)
                obs = self.process_observation(obs)
                ep_rews.append(rew)
                batch_acts.append(action)
                batch_log_probs.append(log_prob)
                if done:
                    break

            batch_lens.append(ep_t + 1)
            batch_rews.append(ep_rews)

        batch_obs = [torch.tensor(batch_ob, dtype=torch.float) for batch_ob in batch_obs]
        batch_acts = torch.tensor(batch_acts, dtype=torch.float)
        batch_log_probs = torch.tensor(batch_log_probs, dtype=torch.float)
        batch_rtgs = self.compute_rtgs(batch_rews)
        return batch_obs, batch_acts, batch_rews, batch_log_probs, batch_rtgs, batch_lens

    """
    Computes the rewards to go for a batch 
    """
    def compute_rtgs(self, batch_rews):
        batch_rtgs = []
        for ep_reward in batch_rews:
            discount_reward = 0
            for rew in reversed(ep_reward):
                discount_reward = rew + discount_reward * self.gamma
                batch_rtgs.insert(0, discount_reward)
        batch_rtgs = torch.tensor(batch_rtgs, dtype=torch.float)
        return batch_rtgs

    def process_observation(self, observation):
        img = Image.fromarray(observation)
        img = img.resize((84, 84))
        img = img.convert("L")
        img = np.array(img)
        img = torch.from_numpy(img)
        img = img.unsqueeze(0)
        img = img.unsqueeze(0)
        img = img / 255.0
        img = img.to(self.device)
        return img

    def save(self, episode):
        torch.save(self.critic.state_dict(), "./model/ppo_critic{}.pth".format(episode))
        torch.save(self.actor.state_dict(), "./model/ppo_actor{}.pth".format(episode))

    def load(self, episode):
        self.critic.load_state_dict(torch.load("./model/ppo_critic{}.pth".format(episode)))
        self.actor.load_state_dict(torch.load("./model/ppo_actor{}.pth".format(episode)))


env = gym.make('Breakout-v4')
model = PPO(env)
model.learn(10000)






