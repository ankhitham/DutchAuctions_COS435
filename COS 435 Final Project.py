# -*- coding: utf-8 -*-
"""
We took inspiration from the following code repositories and papers when writing our own code below:

OpenAI Gym: https://github.com/openai/gym

General Auction Class Construction: https://github.com/ChuaCheowHuan/gym-continuousDoubleAuction/blob/master/gym_continuousDoubleAuction/envs/continuousDoubleAuction_env.py

Stable Baselines-3 PPO Documentation: https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html

Loss Callback: https://github.com/DLR-RM/stable-baselines3/issues/1888; https://stable-baselines3.readthedocs.io/en/master/guide/callbacks.html


Design choices:
1) Draw from uniform distribution: https://www.cs.cornell.edu/home/kleinber/networks-book/networks-book-ch09.pdf

2) Bidder probability of acceptance (sigmoid): https://milgrom.people.stanford.edu/wp-content/uploads/1989/07/Auctions-and-Bidding-Primer.pdf

3) Poisson distribution arrival: https://arxiv.org/pdf/0712.1962
"""

!pip install gymnasium stable-baselines3 shimmy

import numpy as np
import json
import os
import gymnasium as gym
from gymnasium import spaces
from scipy import stats
from scipy.stats import ttest_ind
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import BaseCallback
import matplotlib.pyplot as plt
import pandas as pd

MAX_BIDDER_COUNT = 20
MAX_PRICE = 100.0
MIN_VALUATION = 20.0
MAX_COST = 0.2
BETA = 8.0
LAMBDA_COST = 0.5
LEFTOVER_PENALTY = 20.0
MAX_TIMESTEP = 60

class DutchAuctionEnv(gym.Env):
    def __init__(self,
                 max_bidders=MAX_BIDDER_COUNT,
                 start_price=MAX_PRICE,
                 min_value= MIN_VALUATION,
                 max_cost=MAX_COST,
                 max_timestep=MAX_TIMESTEP,
                 min_drop=0.01,
                 max_drop=0.10,
                 seed=None):
        super().__init__()
        self.max_bidders = max_bidders
        self.start_price = start_price
        self.min_value = min_value
        self.max_cost = max_cost
        self.max_timestep = max_timestep
        self.min_drop = min_drop
        self.max_drop = max_drop
        self.action_space = spaces.Box(low=min_drop, high=max_drop, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(0.0, 1.0, shape=(6,), dtype=np.float32)
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.t = 0
        self.last_drop = 0.0
        self.price = self.start_price
        self.N = max(1, np.random.poisson(self.max_bidders))
        self.v = np.random.uniform(self.min_value, self.start_price, size=self.N)
        self.c = np.random.uniform(0.0, self.max_cost, size=self.N)
        self.beta = np.random.uniform(0.5, BETA, size=self.N)

        self.done = False
        return self._get_obs(), {}

    def _get_obs(self):
        U = self.v - self.price - self.c * self.t
        almost = np.sum(U >= -0.05*self.start_price)
        return np.array([
            self.price/self.start_price,
            self.t/self.max_timestep,
            self.N/self.max_bidders,
            self.v.mean()/self.start_price,
            almost/self.N,
            self.last_drop
        ], dtype=np.float32)

    def step(self, action):
        assert not self.done
        self.last_drop = float(np.clip(action, self.min_drop, self.max_drop))
        self.price = max(self.min_value, self.price*(1-self.last_drop))
        self.t += 1

        U = self.v - self.price - self.c*self.t
        buy_prob = 1/(1+np.exp(-self.beta*U))
        draws = np.random.rand(self.N)
        buys = draws < buy_prob

        if buys.any():
            self.done = True
            reward = self.price - LAMBDA_COST*self.t
        elif self.t >= self.max_timestep:
            self.done = True
            reward = -LEFTOVER_PENALTY
        else:
            reward = -LAMBDA_COST

        info = {
            'ticks_to_sale': self.t if self.done else None,
            'final_price': self.price if self.done else None,
        }
        return self._get_obs(), reward, self.done, False, info

class LossCallback(BaseCallback):
    def __init__(self):
        super().__init__(verbose=0)
        self.losses = []

    def _on_step(self):
        self.losses.append(self.model.logger.name_to_value.get('train/loss'))
        return True

    def plot(self):
        plt.figure()
        plt.plot(self.losses)
        plt.title("PPO Training Loss")
        plt.xlabel("Update")
        plt.ylabel("Loss")
        plt.tight_layout()
        plt.show()

def train_seller(seed, timesteps):
    env = DummyVecEnv([lambda: DutchAuctionEnv(seed=seed+i) for i in range(4)])
    env = VecNormalize(env, norm_obs=True, norm_reward=False)
    loss_cb = LossCallback()
    model = PPO(
        "MlpPolicy", env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=256,
        ent_coef=0.005,
        gamma=0.99,
        seed=seed,
        verbose=1
    )
    model.learn(total_timesteps=timesteps, callback=loss_cb)
    return model, env, loss_cb

def evaluate_policy(model, env, n_eval=1000):
    rews, ticks, prices = [], [], []
    for _ in range(n_eval):
        obs, _ = env.reset()
        done = False
        total_r = 0
        while not done:
            act, _ = model.predict(obs, deterministic=True)
            obs, r, done, _, info = env.step(act)
            total_r += r
        rews.append(total_r)
        ticks.append(info['ticks_to_sale'])
        prices.append(info['final_price'])
    return np.array(rews), np.array(ticks), np.array(prices)

def evaluate_baseline(g, n_eval=1000):
    rews, ticks, prices = [], [], []
    for seed in range(5):
        env = DutchAuctionEnv(seed=seed)
        for _ in range(n_eval//5):
            obs, _ = env.reset()
            done = False
            total_r = 0
            while not done:
                obs, r, done, _, info = env.step([g])
                total_r += r
            rews.append(total_r)
            ticks.append(info['ticks_to_sale'])
            prices.append(info['final_price'])
    return np.array(rews), np.array(ticks), np.array(prices)

model, env, loss_cb = train_seller(seed=0, timesteps=200_000)
loss_cb.plot()
rl_rews, rl_ticks, rl_prices = evaluate_policy(model, env.envs[0])

grid = np.linspace(0.01, 0.10, 10)
best_mean, best_g = -np.inf, None
for g in grid:
    br, bt, bp = evaluate_baseline(g)
    m = br.mean()
    if m > best_mean:
        best_mean, best_g = m, g
bl_rews, bl_ticks, bl_prices = evaluate_baseline(best_g)

print(pd.DataFrame([
    ['RL', rl_rews.mean(), rl_rews.std(), rl_ticks.mean(), rl_prices.mean()],
    ['Baseline', bl_rews.mean(), bl_rews.std(), bl_ticks.mean(), bl_prices.mean()]
], columns=['policy','mean','std','ticks','price']))

t, p = ttest_ind(rl_rews, bl_rews, equal_var=False)
print("p-value:", p)

plt.figure()
plt.hist(rl_rews, bins=30, label='PPO')
plt.hist(bl_rews, bins=30, label='Baseline g*=0.02',)
plt.title("Total Profit per Auction")
plt.xlabel("Profit")
plt.ylabel("Number of Auctions")
plt.legend()
plt.show()

plt.figure()
plt.boxplot([rl_ticks, bl_ticks], labels=['PPO', 'Baseline'])
plt.title("Number of Timesteps per Auction")
plt.ylabel("Number of Timesteps")
plt.show()

plt.figure()
plt.hist(rl_prices, bins=30, label='PPO')
plt.hist(bl_prices, bins=30, label='Baseline')
plt.axvline(np.mean(rl_prices), label='PPO mean')
plt.axvline(np.mean(bl_prices), label='Baseline mean', color = 'orange')
plt.title("Final Sale Price per Auction")
plt.xlabel("Final Sale Price")
plt.ylabel("Number of Auctions")
plt.legend()
plt.show()
