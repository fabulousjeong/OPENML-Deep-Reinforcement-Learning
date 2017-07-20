#참조: https://hunkim.github.io/ml/

import tensorflow
import gym
import numpy as np
import matplotlib.pyplot as plt
from gym.envs.registration import register
import random as pr

register(
    id = 'FrozenLake-v3',
    entry_point = 'gym.envs.toy_text:FrozenLakeEnv',
    kwargs = {
        'map_name': '4x4',
        'is_slippery': False
    }
)

env = gym.make('FrozenLake-v3')
# Q-table 초기화
Q = np.zeros([env.observation_space.n, env.action_space.n])

# discount factor
dis = .99

# learning 횟수
num_episodes = 2000

# 학습 시 reward 저장
rList = []

for i in range(num_episodes):
    # env 리셋
    state = env.reset()
    rAll = 0
    done = False
    #decaying E-greed
    e = 1./((i//100)+1)
    # Q-table Learning algorithm
    if i%100 is 0:
        print('i:',i)
    while not done:

        # Choose an action by e greedy
        if np.random.rand(1) < e:
            action = env.action_space.sample()
        else:
            action = np.argmax(Q[state, :])


        # add Random noise
        # action = np.argmax(Q[state, :]+ np.random.randn(1, env.action_space.n)/(i + 1))
        # new_state, reward 업데이트
        new_state, reward, done, _ = env.step(action)

        # update Q-table
        Q[state, action] = reward + dis*np.max(Q[new_state, :])

        rAll += reward
        state = new_state

    rList.append(rAll)


print('success rate: ', str(sum(rList)/num_episodes))
print('Final Q-table values')
print(Q)
plt.bar(range(len(rList)), rList, color = 'blue')
plt.show()
