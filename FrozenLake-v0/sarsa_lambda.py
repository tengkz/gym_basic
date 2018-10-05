#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on 2018.10.05
@author: tengkz@gmail.com
TODO:
    1. how to initialize Q (as low as possible)
    2. how to change epsilon
    3. how to change alpha
    4. how to set LAMBDA (bigger for long trace, smaller for short one)
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
Q = np.random.uniform(0.0,0.05,size=[state_space_n,action_space_n])
N = np.zeros(shape=[state_space_n,action_space_n],dtype=np.int32)
E = np.zeros(shape=[state_space_n,action_space_n],dtype=np.float32)
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

def pick_alpha(episode):
    return 1.0/(episode/100.0+1.0)

GAMMA = 0.95
QUEUE_SIZE = 100
LAMBDA = 0.5
EPISODE = 10000

queue = Queue.Queue(QUEUE_SIZE)
result = []

for episode in range(EPISODE):
    E.fill(0.0)
    state = env.reset()
    epsilon = pick_epsilon(episode)
    alpha = pick_alpha(episode)
    action = pick_action_epsilon_greedy(state,epsilon)
    while True:
        state_next,reward,done,info = env.step(action)
        N[state,action] += 1
        action_next = pick_action_epsilon_greedy(state_next,epsilon)
        delta = reward+GAMMA*Q[state_next,action_next]-Q[state,action]
        E[state,action] += 1.0
        for s in range(state_space_n):
            for a in range(action_space_n):
                Q[s,a]+=alpha*delta*E[s,a]
                E[s,a]*=GAMMA*LAMBDA
        state = state_next
        action = action_next
        if done:
            if episode>=QUEUE_SIZE:
                queue.get()
            queue.put(reward)
            break
    if episode%1000 == 0:
        print 'episode=%d, epsilon=%f' % (episode,epsilon)
    result.append(1.0*sum(queue.queue)/QUEUE_SIZE)

plt.plot(result)
plt.show()
print sum(result)/EPISODE