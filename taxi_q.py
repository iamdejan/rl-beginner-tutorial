import os

import gymnasium as gym
import numpy as np
from matplotlib import pyplot as plt

PREFIX = r"taxi_v3"
Q_NPY_FILE_NAME = PREFIX + ".npy"
SEED = 1770648564


def run(episodes: int, is_training: bool = True, render: bool = False):
    env = gym.make("Taxi-v3", render_mode="human" if render else None)

    if is_training:
        q = np.zeros(shape=(env.observation_space.n, env.action_space.n))
    else:
        q = np.load(Q_NPY_FILE_NAME)

    learning_rate_a = 0.9  # alpha or learning rate
    discount_factor_g = 0.9  # gamma or discount rate

    epsilon = 1  # 1 = 100% random actions
    epsilon_decay_rate = 0.000005  # epsilon decay rate
    rng = np.random.default_rng()

    rewards_per_episode = np.zeros(episodes)

    for i in range(episodes):
        state = env.reset(seed=SEED)[0]
        terminated = False  # True when collide or reached goal
        truncated = False  # True when actions > 200

        rewards = 0
        while not terminated and not truncated:
            if is_training and rng.random() < epsilon:
                action = env.action_space.sample()  # 0 = move left, 1 = move down, 2 = move right, 3 = move up
            else:
                action = np.argmax(q[state, :])

            new_state, reward, terminated, truncated, _ = env.step(action)
            rewards += reward

            if is_training:
                q[state, action] = q[state, action] + learning_rate_a * (
                    reward + discount_factor_g * np.max(q[new_state, :]) - q[state, action]
                )
                print(f"Episode {i} -> q[{state}, {action}] = {q[state, action]}")

            state = new_state

        epsilon = max(epsilon - epsilon_decay_rate, 0)
        if epsilon == 0:
            learning_rate_a = 0.0001

        rewards_per_episode[i] = rewards

        print(f"Episode {i} ends\n")

    env.close()

    sum_rewards = np.zeros(episodes)
    for t in range(episodes):
        sum_rewards[t] = np.sum(rewards_per_episode[max(0, t - 100) : (t + 1)])
    plt.plot(sum_rewards)

    if is_training:
        plot_file_name = PREFIX + "_train.png"
    else:
        plot_file_name = PREFIX + "_test.png"
    plt.savefig(plot_file_name)

    if is_training:
        np.save(Q_NPY_FILE_NAME, q)


def main():
    np.random.seed(SEED)

    episodes: int = int(os.getenv("EPISODES"))
    is_training: bool = os.getenv("IS_TRAINING") == "true"
    render: bool = os.getenv("RENDER") == "true"

    run(episodes=episodes, is_training=is_training, render=render)


if __name__ == "__main__":
    main()
