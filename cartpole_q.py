import os

import gymnasium as gym
import numpy as np
from matplotlib import pyplot as plt

PREFIX = r"cartpole"
Q_NPY_FILE_NAME = PREFIX + ".npy"
SEED = 1770648564
SEGMENTS = 10


def run(is_training: bool = True, render: bool = False):
    env = gym.make("CartPole-v1", render_mode="human" if render else None)

    pos_space = np.linspace(-2.4, 2.4, SEGMENTS)
    vel_space = np.linspace(-5, 5, SEGMENTS)
    ang_space = np.linspace(-0.21, 0.21, SEGMENTS)
    ang_vel_space = np.linspace(-4, 4, SEGMENTS)

    if is_training:
        q = np.zeros(
            shape=(
                len(pos_space) + 1,
                len(vel_space) + 1,
                len(ang_space) + 1,
                len(ang_vel_space) + 1,
                env.action_space.n,
            )
        )
    else:
        q = np.load(Q_NPY_FILE_NAME)

    learning_rate_a = 0.2  # alpha or learning rate
    discount_factor_g = 0.99  # gamma or discount rate

    epsilon = 1  # 1 = 100% random actions
    epsilon_decay_rate = 0.00001  # epsilon decay rate
    rng = np.random.default_rng()

    rewards_per_episode = []

    i = 0

    while True:
        state = env.reset(seed=SEED)[0]
        state_p = np.digitize(state[0], pos_space)
        state_v = np.digitize(state[1], vel_space)
        state_a = np.digitize(state[2], ang_space)
        state_av = np.digitize(state[3], ang_vel_space)

        terminated = False

        rewards = 0

        while not terminated and rewards < 10000:
            if is_training and rng.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(q[state_p, state_v, state_a, state_av, :])

            new_state, reward, terminated, _, _ = env.step(action)
            new_state_p = np.digitize(new_state[0], pos_space)
            new_state_v = np.digitize(new_state[1], vel_space)
            new_state_a = np.digitize(new_state[2], ang_space)
            new_state_av = np.digitize(new_state[3], ang_vel_space)
            if is_training:
                q[state_p, state_v, state_a, state_av, action] = q[state_p, state_v, state_a, state_av, action] + learning_rate_a * (
                    reward
                    + discount_factor_g * np.max(q[new_state_p, new_state_v, new_state_a, new_state_av, :])
                    - q[state_p, state_v, state_a, state_av, action]
                )

            state = new_state
            state_p = new_state_p
            state_v = new_state_v
            state_a = new_state_a
            state_av = new_state_av

            rewards += reward

        epsilon = max(epsilon - epsilon_decay_rate, 0)
        if epsilon == 0:
            learning_rate_a = 0.0001

        rewards_per_episode.append(rewards)
        mean_rewards = np.mean(rewards_per_episode[len(rewards_per_episode) - 100 :])
        if is_training and i % 100 == 0:
            print(f"Episode: {i} {rewards}  Epsilon: {epsilon:0.2f}  Mean rewards: {mean_rewards:0.1f}")

        if mean_rewards > 1000:
            break

        i += 1

    env.close()

    mean_rewards = []
    for t in range(i):
        mean_rewards.append(np.mean(rewards_per_episode[max(0, t - 100) : (t + 1)]))
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

    is_training: bool = "true" == os.getenv("IS_TRAINING")
    render: bool = "true" == os.getenv("RENDER")

    run(is_training=is_training, render=render)


if __name__ == "__main__":
    main()
