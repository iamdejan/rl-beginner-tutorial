import os

import gymnasium as gym
import numpy as np
from matplotlib import pyplot as plt

PREFIX = r"mountain_car_cont_alt"
Q_NPY_FILE_NAME = PREFIX + ".npy"
SEED = 1770648564


def run(is_training: bool = True, render: bool = False):
    env = gym.make("MountainCarContinuous-v0", render_mode="human" if render else None)

    # hyperparameters
    learning_rate_a = 0.9999  # alpha or learning rate
    discount_factor_g = 0.9  # gamma or discount rate
    epsilon = 1  # 1 = 100% random actions
    epsilon_decay_rate = 0.0005  # epsilon decay rate
    epsilon_min = 0.05

    # segment count
    pos_divisions = 20
    vel_divisions = 20
    act_divisions = 10

    # Divide observation space into discrete elements
    pos_space = np.linspace(env.observation_space.low[0], env.observation_space.high[0], pos_divisions)
    vel_space = np.linspace(env.observation_space.low[1], env.observation_space.high[1], vel_divisions)

    # Divide action space into discrete elements
    act_space = np.linspace(env.action_space.low[0], env.action_space.high[0], act_divisions, endpoint=False)  # Between [-1, 1]

    if is_training:
        q = np.zeros(
            shape=(
                len(pos_space) + 1,
                len(vel_space) + 1,
                len(act_space) + 1,
            )
        )
    else:
        q = np.load(Q_NPY_FILE_NAME)

    best_reward = -999999  # track best reward
    rewards_per_episode = []  # list to store rewards for each episode
    epsilon_history = []  # List to keep track of epsilon decay
    i = 0  # episode counter

    while True:
        state = env.reset(seed=SEED)[0]
        state_p = np.digitize(state[0], pos_space)
        state_v = np.digitize(state[1], vel_space)

        terminated = False

        rewards = 0
        steps = 0
        while not terminated and steps < 5000:
            random_number = np.random.rand()
            if is_training and random_number < epsilon:
                action = env.action_space.sample()[0]
                action_idx = np.digitize(action, act_space)
            else:
                action_idx = np.argmax(q[state_p, state_v, :])
                action = act_space[action_idx - 1]

            # Execute action
            new_state, reward, terminated, _, _ = env.step([action])

            new_state_p = np.digitize(new_state[0], pos_space)
            new_state_v = np.digitize(new_state[1], vel_space)

            # Update Q table
            if is_training:
                q[state_p, state_v, action_idx] += learning_rate_a * (
                    reward + discount_factor_g * np.max(q[new_state_p, new_state_v, :]) - q[state_p, state_v, action_idx]
                )

            # Set state to new state
            state = new_state
            state_p = new_state_p
            state_v = new_state_v

            rewards += reward
            steps += 1

        if rewards > best_reward:
            best_reward = rewards
            if is_training:
                np.save(Q_NPY_FILE_NAME, q)

        rewards_per_episode.append(rewards)

        print(f"Episode {i}, rewards = {rewards}")

        if rewards > env.spec.reward_threshold:
            break

        epsilon_history.append(epsilon)
        epsilon = max(epsilon - epsilon_decay_rate, epsilon_min)

        if not is_training:
            # it will never improve
            break

        i += 1

    env.close()

    mean_rewards = []
    for t in range(i):
        mean_rewards.append(np.mean(rewards_per_episode[max(0, t - 100) : (t + 1)]))
    plt.plot(mean_rewards)

    # draw plot only if it's training
    if is_training:
        plot_file_name = PREFIX + "_train.png"
        plt.savefig(plot_file_name)
        plt.clf()

        plt.plot(epsilon_history)
        plot_file_name = PREFIX + "_epsilon_history_train.png"
        plt.savefig(plot_file_name)


def main():
    np.random.seed(SEED)

    is_training: bool = "true" == os.getenv("IS_TRAINING")
    render: bool = "true" == os.getenv("RENDER")

    run(is_training=is_training, render=render)


if __name__ == "__main__":
    main()
