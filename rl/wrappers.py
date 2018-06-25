#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gym Wrappers
@author: thomas
"""
import gym
import numpy as np
from scipy.stats import multivariate_normal
import sklearn.preprocessing

class ObservationRewardWrapper(gym.Wrapper):
    ''' My own base class - allows for both observation and reward modification '''
    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        return self.observation(observation), self.reward(reward), done, info

    def reset(self):
        observation = self.env.reset()
        return self.observation(observation)

    def observation(self, observation):
        return observation 

    def reward(self, reward):
        return reward

def get_name(env):
    while True:
        if hasattr(env,'_spec'):
            name = env._spec.id
            break
        elif hasattr(env,'spec'): 
            name = env.spec.id
            break
        else:
            env = env.env
    return name

class NormalizeWrapper(ObservationRewardWrapper):
    ''' normalizes the input data range '''
    def __init__(self, env):
        ObservationRewardWrapper.__init__(self, env)
        #self.name = get_name(env)    
        observation_examples = np.array([env.observation_space.sample() for x in range(10000)])
        self.scaler = sklearn.preprocessing.StandardScaler()
        self.scaler.fit(observation_examples)
        
    def observation(self, observation):
        return self.scaler.transform([observation])[0]

class ScaleRewardWrapper(ObservationRewardWrapper):
    
    def __init__(self, env):
        ObservationRewardWrapper.__init__(self, env)
        self.name = get_name(env)

    def reward(self, reward):
        """ Rescale reward """
        if 'Pendulum' in self.name:
            return np.float32(reward/1000.0)
        #elif 'MountainCarContinuous' in self.name:
        #    return np.float32(reward/500.0)
        elif 'Lunarlander' in self.name:
            return np.float32(reward/250.0)
        elif 'CartPole' in self.name:
            return reward/250.0
        elif 'MountainCar' in self.name:
            return reward/250.0
        elif 'Acrobot' in self.name:
            return reward/250.0
        else:
            return reward
          
class ReparametrizeWrapper(ObservationRewardWrapper):

    def __init__(self, env):
        ObservationRewardWrapper.__init__(self, env)
        self.name = get_name(env)

    def step(self, action):
        observation, reward, terminal, info = self.env.step(action)
        return self.observation(observation), self.reward(reward,terminal), terminal, info

    def reward(self,r,terminal):
        if 'CartPole' in self.name:
            if terminal:
                r = -1
            else:
                r = 0.005
        elif 'MountainCar' in self.name:
            if terminal:
                r = 1
            else:
                r = -0.005
        elif 'Acrobot' in self.name:
            if terminal:
                r = 1
            else:
                r = -0.005
        return r

class PILCOWrapper(ObservationRewardWrapper):

    def __init__(self, env):
        ObservationRewardWrapper.__init__(self, env)
        self.name = get_name(env)

    def step(self, action):
        observation, reward, terminal, info = self.env.step(action)
        return self.observation(observation), self.reward(observation), terminal, info

    def reward(self,s):
        if 'CartPole' in self.name:
            target = np.array([0.0,0.0,0.0,0.0])
        elif 'Acrobot' in self.name:
            target = np.array([1.0])
            s = -np.cos(s[0]) - np.cos(s[1] + s[0])
        elif 'MountainCar' in self.name:
            target = np.array([0.5])
            s = s[0]
        elif 'Pendulum' in self.name:
            target = np.array([0.0,0.0])
        else:
            raise ValueError('no PILCO reward mofication for this game')
        return 1 - multivariate_normal.pdf(s,mean=target)
    
class ClipRewardWrapper(ObservationRewardWrapper):
    
    def __init__(self, env):
        ObservationRewardWrapper.__init__(self, env)

    def reward(self, reward):
        """Bin reward to {+1, 0, -1} by its sign."""
        return np.sign(reward)
    
class ScaledObservationWrapper(ObservationRewardWrapper):
    def __init__(self, env):
        ObservationRewardWrapper.__init__(self, env)

    def observation(self, observation):
        return np.array(observation).astype(np.float32) / 255.0

