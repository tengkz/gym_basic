#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on 2018.10.05
@author: tengkz@gmail.com
TODO:
    1. how to initialize Q
    2. how to change epsilon
"""

import gym
import numpy as np
import Queue
import matplotlib.pyplot as plt

env = gym.make('FrozenLake-v0')
env.reset()

# q lookup table
state_space_n = env.observation_space.n
action_space_n = env.action_space.n
# random strategy
# exp(GAMMA,state_space_n)~=0.5
Q = np.random.uniform(0.0,0.5,size=[state_space_n,action_space_n])
N = np.zeros(shape=[state_space_n,action_space_n],dtype=np.int32)
# specific for FrozenLake-v0
Q[15,:] = 0.0

def pick_action_greedy(state):
    return np.argmax(Q[state,:])

def pick_action_epsilon_greedy(state,epsilon):
    if np.random.uniform(0.0,1.0,1)<epsilon:
        return env.action_space.sample()
    else:
        return pick_action_greedy(state)

def pick_epsilon(episode):
    return 1.0/(episode/100.0+1.0)

GAMMA = 0.95
QUEUE_SIZE = 100
EPISODE = 10000

queue = Queue.Queue(QUEUE_SIZE)
result = []

for episode in range(EPISODE):
    state = env.reset()
    epsilon = pick_epsilon(episode)
    sequence = []
    while True:
        action = pick_action_epsilon_greedy(state,epsilon)
        state_next,reward,done,info = env.step(action)
        sequence.append([state,action,reward])
        state = state_next
        if done:
            if episode>=QUEUE_SIZE:
                queue.get()
            queue.put(reward)
            break
    length = len(sequence)
    G = reward
    for i in range(length-1,-1,-1):
        state,action,r = sequence[i]
        N[state,action] += 1
        Q[state,action] += 1.0/N[state,action]*(G-Q[state,action])
        G = G*GAMMA+r
    precision = 1.0*sum(queue.queue)/QUEUE_SIZE)
    if episode%1000 == 0:
        print 'episode=%d, epsilon=%f, precision=%f' % (episode,epsilon,precision)
    result.append(precision)

plt.plot(result)
plt.show()
print sum(result)/EPISODE
