from enum import Enum

from DeepQLearning.dqn import Agent
from environment import Environment, Action
from utils.plot import plotLearning
from utils.read_maze import load_maze

import numpy as np


def main():
    agent = Agent(gamma=.99, epsilon=1.0, batch_size=64, n_actions=5, epsilon_end=.01, input_dims=[18], lr=.001)
    load_maze()
    env = Environment()
    scores, epsilon_history = [], []
    number_games = 200

    for i in range(number_games):
        score = 0
        done = False
        observation = env.reset
        while not done:
            #print(observation)

            action = agent.choose_action(observation)
            observation_, reward, done = env.step(action)
            env.update_total_reward(reward)
            score += reward
            agent.store_transition(observation, action, reward, observation_, done)
            agent.learn()
            observation = observation_
            #print(score)
        scores.append(score)
        epsilon_history.append(agent.epsilon)

        avg_score = np.mean(scores[-100:])

        print('episode ', i, "score %.2f" % score, 'average score %.2f' % avg_score, 'epsilon %.2f' % agent.epsilon)

    x = [i + 1 for i in range(number_games)]
    filename = "dynamic_maze.png"
    plotLearning(x, scores, epsilon_history, filename)


if __name__ == "__main__":
    main()
