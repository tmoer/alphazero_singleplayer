# -*- coding: utf-8 -*-
"""
Custom game generation function
@author: thomas
"""
import gym
import numpy as np
from .wrappers import NormalizeWrapper,ReparametrizeWrapper,PILCOWrapper,ScaleRewardWrapper,ClipRewardWrapper,ScaledObservationWrapper

# Register deterministic FrozenLakes
from gym.envs.registration import register
register(
    id='FrozenLakeNotSlippery-v0',
    entry_point='gym.envs.toy_text:FrozenLakeEnv',
    kwargs={'map_name' : '4x4', 'is_slippery': False},
    max_episode_steps=100,
    reward_threshold=0.78, # optimum = .8196
)
register(
    id='FrozenLakeNotSlippery-v1',
    entry_point='gym.envs.toy_text:FrozenLakeEnv',
    kwargs={'map_name' : '8x8', 'is_slippery': False},
    max_episode_steps=100,
    reward_threshold=0.78, # optimum = .8196
)

def get_base_env(env):
    ''' removes all wrappers '''
    while hasattr(env,'env'):
        env = env.env
    return env

def is_atari_game(env):
    ''' Verify whether game uses the Arcade Learning Environment '''
    env = get_base_env(env)
    return hasattr(env,'ale')

def make_game(game):
    ''' Modifications to Env '''
    name,version = game.rsplit('-',1)
    if len(version) > 2:
        modify = version[2:]
        game = name + '-' + version[:2]
    else:
        modify = ''

    print('Making game {}'.format(game))        
    env = gym.make(game)
    # remove timelimit wrapper
    if type(env) == gym.wrappers.time_limit.TimeLimit:
        env = env.env
    
    if is_atari_game(env):
        env = prepare_atari_env(env)
    else:
        env = prepare_control_env(env,game,modify)
    return env

def prepare_control_env(env,game,modify):
    if 'n' in modify and type(env.observation_space) == gym.spaces.Box:
        print('Normalizing input space')        
        env = NormalizeWrapper(env)        
    if 'r' in modify:
        print('Reparametrizing the reward function')        
        env = ReparametrizeWrapper(env)
    if 'p' in modify:
        env = PILCOWrapper(env)
    if 's' in modify:
        print('Rescaled the reward function')        
        env = ScaleRewardWrapper(env)
    
    if 'CartPole' in game:
        env.observation_space = gym.spaces.Box(np.array([-4.8,-10,-4.8,-10]),np.array([4.8,10,4.8,10]))        
    return env

def prepare_atari_env(Env,frame_skip=3,repeat_action_prob=0.0,reward_clip=True):
    ''' Initialize an Atari environment '''
    env = get_base_env(Env)
    env.ale.setFloat('repeat_action_probability'.encode('utf-8'), repeat_action_prob)
    env.frame_skip = frame_skip
    Env = ScaledObservationWrapper(Env)
    if reward_clip:
        Env = ClipRewardWrapper(Env)
    return Env
