import os

import gymnasium as gym
import numpy as np
from matplotlib import pyplot as plt

PREFIX = r"mountain_car"
Q_NPY_FILE_NAME = PREFIX + ".npy"
SEED = 1770648564
SEGMENTS = 20


def run(episodes: int, is_training: bool = True, render: bool = False):
    env = gym.make("MountainCar-v0", render_mode="human" if render else None)

    # Divide positions and velocity into segments
    pos_space = np.linspace(env.observation_space.low[0], env.observation_space.high[0], SEGMENTS)  # between -1.2 and 0.6
    vel_space = np.linspace(env.observation_space.low[1], env.observation_space.high[1], SEGMENTS)  # between -0.07 and 0.7

    if is_training:
        q = np.zeros(shape=(len(pos_space), len(vel_space), env.action_space.n))
    else:
        q = np.load(Q_NPY_FILE_NAME)

    learning_rate_a = 0.9  # alpha or learning rate
    discount_factor_g = 0.9  # gamma or discount rate

    epsilon = 1  # 1 = 100% random actions
    epsilon_decay_rate = 2 / episodes  # epsilon decay rate
    rng = np.random.default_rng()

    rewards_per_episode = np.zeros(episodes)

    for i in range(episodes):
        state = env.reset(seed=SEED)[0]
        state_p = np.digitize(state[0], pos_space)
        state_v = np.digitize(state[1], vel_space)

        terminated = False

        rewards = 0

        while not terminated and rewards > -1000:
            if is_training and rng.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(q[state_p, state_v, :])

            new_state, reward, terminated, _, _ = env.step(action)
            new_state_p = np.digitize(new_state[0], pos_space)
            new_state_v = np.digitize(new_state[1], vel_space)
            if is_training:
                q[state_p, state_v, action] = q[state_p, state_v, action] + learning_rate_a * (
                    reward + discount_factor_g * np.max(q[new_state_p, new_state_v, :]) - q[state_p, state_v, action]
                )
                print(f"Episode {i} -> q[{state_p}, {state_v}, {action}] = {q[state_p, state_v, action]}")

            state = new_state
            state_p = new_state_p
            state_v = new_state_v
            rewards += reward

        epsilon = max(epsilon - epsilon_decay_rate, 0)
        if epsilon == 0:
            learning_rate_a = 0.0001

        rewards_per_episode[i] = rewards

        print(f"Episode {i} ends\n")

    env.close()

    mean_rewards = np.zeros(episodes)
    for t in range(episodes):
        mean_rewards[t] = np.mean(rewards_per_episode[max(0, t - 100) : (t + 1)])
    plt.plot(mean_rewards)

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
