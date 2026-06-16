

import gymnasium as gym
import numpy as np
from collections import defaultdict
import pickle

env = gym.make('Blackjack-v1', natural=False, sab=False)

alpha = 0.95
gamma = 0.99

epsilon = 1.0
epsilon_min = 0.1
epsilon_decay = 0.96

q_table = defaultdict(lambda: np.zeros(env.action_space.n))

wins = 0 #-> Contagem de vitórias

for episodio in range(100000000):

    state, info = env.reset()
    done = False

    while not done:

        # Escolha da ação (ε-greedy)
        if np.random.random() < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(q_table[state])

        next_state, reward, terminated, truncated, info = env.step(action)

        done = terminated or truncated

        # Valor futuro
        if done:
            best_next = 0
        else:
            best_next = np.max(q_table[next_state])

        # Target do Q-Learning
        target = reward + gamma * best_next

        # Atualização da Q-table
        q_table[state][action] += alpha * (
            target - q_table[state][action])

        state = next_state

        if done and reward == 1:
            wins += 1

    # Decaimento do epsilon ao final do episódio
    epsilon = max(epsilon_min, epsilon * epsilon_decay)

print(f"Vitórias: {wins}")
print(f"Taxa de vitória: {wins/100000000*100:.2f}%")


with open("blackjack_qtable.pkl", "wb") as treinoBlackJack:
    pickle.dump(dict(q_table),treinoBlackJack)

'''
A observação consiste em uma tupla de 3 elementos contendo: a soma atual do jogador, o valor da única carta visível do dealer (de 1 a 10, onde 1 é o ás) e se o jogador possui um ás válido (0 ou 1).'''
