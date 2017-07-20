#참조: https://hunkim.github.io/ml/

import tensorflow
import gym
import numpy as np
import matplotlib.pyplot as plt
from gym.envs.registration import register
import random as pr


def rargmax(vector):
    # amax는 최고 값을 가지는 index를 무작위로 선택한다.
    m = np.amax(vector)
    indices = np.nonzero(vector == m)[0]
    return pr.choice(indices)

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
# learning 횟수
num_episodes = 2000
# 학습 시 reward 저장
rList = []


for i in range(num_episodes):
    # env 리셋
    state = env.reset()
    rAll = 0
    done = False
    # Q-table Learning algorithm
    while not done:
        action = rargmax(Q[state, :])

        # new_state, reward 업데이트
        new_state, reward, done, _ = env.step(action)

        # update Q-table
        Q[state, action] = reward + np.max(Q[new_state, :])

        rAll += reward
        state = new_state



    rList.append(rAll)

print('success rate: ', str(sum(rList)/num_episodes))
print('Final Q-table values')
print(Q)
plt.bar(range(len(rList)), rList, color = 'blue')
plt.show()
