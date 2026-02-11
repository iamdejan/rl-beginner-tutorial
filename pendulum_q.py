import os

import gymnasium as gym
import numpy as np

PREFIX = r"pendulum"
Q_NPY_FILE_NAME = PREFIX + ".npy"
SEED = 1770648564
SEGMENTS = 15


def run(is_training: bool = True, render: bool = False):
    env = gym.make("Pendulum-v1", render_mode="human" if render else None)

    # hyperparameters
    learning_rate_a = 0.1  # alpha or learning rate
    discount_factor_g = 0.9  # gamma or discount rate
    epsilon = 1  # 1 = 100% random actions
    epsilon_decay_rate = 0.0005  # epsilon decay rate
    epsilon_min = 0.05

    # Divide observation space into discrete elements
    x = np.linspace(env.observation_space.low[0], env.observation_space.high[0], SEGMENTS)
    y = np.linspace(env.observation_space.low[1], env.observation_space.high[1], SEGMENTS)
    av = np.linspace(env.observation_space.low[2], env.observation_space.high[2], SEGMENTS)  # angular velocity

    # Divide action space into discrete elements
    a = np.linspace(env.action_space.low[0], env.action_space.high[0], SEGMENTS)

    if is_training:
        q = np.zeros(
            shape=(
                len(x) + 1,
                len(y) + 1,
                len(av) + 1,
                len(a) + 1,
            )
        )
    else:
        q = np.load(Q_NPY_FILE_NAME)

    best_reward = -99999
    rewards_per_episode = []
    i = 0

    while True:
        state = env.reset(seed=SEED)[0]

        s_i0 = np.digitize(state[0], x)
        s_i1 = np.digitize(state[1], y)
        s_i2 = np.digitize(state[2], av)

        rewards = 0.0
        steps = 0

        while steps < 1000 or is_training is False:
            if is_training and np.random.rand() < epsilon:
                action = env.action_space.sample()[0]
                action_idx = np.digitize(action, a)
            else:
                action_idx = np.argmax(q[s_i0, s_i1, s_i2, :])
                action = a[action_idx - 1]

            # Take action
            new_state, reward, _, _, _ = env.step([action])

            # Convert continuous state
            ns_i0 = np.digitize(new_state[0], x)
            ns_i1 = np.digitize(new_state[1], y)
            ns_i2 = np.digitize(new_state[2], av)

            # Update Q table
            if is_training:
                q[s_i0, s_i1, s_i2, action_idx] += learning_rate_a * (
                    reward + discount_factor_g * np.max(q[ns_i0, ns_i1, ns_i2, :]) - q[s_i0, s_i1, s_i2, action_idx]
                )

            # Set state to new state
            state = new_state
            s_i0 = ns_i0
            s_i1 = ns_i1
            s_i2 = ns_i2

            rewards += reward
            steps += 1

        if rewards > best_reward:
            best_reward = rewards
            if is_training:
                np.save(Q_NPY_FILE_NAME, q)

        rewards_per_episode.append(rewards)

        # Print stats
        if is_training and i != 0 and i % 100 == 0:
            mean_reward = np.mean(rewards_per_episode[len(rewards_per_episode) - 100 :])
            print(f"Episode: {i}, Epsilon: {epsilon:0.2f}, Best Reward: {best_reward}, Mean Rewards {mean_reward:0.1f}")
        elif not is_training:
            print(f"Episode: {i}, Rewards: {rewards:0.1f}")

        epsilon = max(epsilon - epsilon_decay_rate, epsilon_min)

        print(f"Episode {i} ends")
        i += 1


def main():
    np.random.seed(SEED)

    is_training: bool = "true" == os.getenv("IS_TRAINING")
    render: bool = "true" == os.getenv("RENDER")

    run(is_training=is_training, render=render)


if __name__ == "__main__":
    main()
