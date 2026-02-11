import os

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np

PREFIX = r"acrobot"
Q_NPY_FILE_NAME = PREFIX + ".npy"
SEED = 1770648564
SEGMENTS = 15


def run(is_training: bool = True, render: bool = False):
    env = gym.make("Acrobot-v1", render_mode="human" if render else None)

    # hyperparameters
    learning_rate_a = 0.1  # alpha or learning rate
    discount_factor_g = 0.9  # gamma or discount rate
    epsilon = 1  # 1 = 100% random actions
    epsilon_decay_rate = 0.0005  # epsilon decay rate
    epsilon_min = 0.05

    # Divide continuous observation space into discrete segments
    th1_cos = np.linspace(env.observation_space.low[0], env.observation_space.high[0], SEGMENTS)
    th1_sin = np.linspace(env.observation_space.low[1], env.observation_space.high[1], SEGMENTS)
    th2_cos = np.linspace(env.observation_space.low[2], env.observation_space.high[2], SEGMENTS)
    th2_sin = np.linspace(env.observation_space.low[3], env.observation_space.high[3], SEGMENTS)
    th1_av = np.linspace(env.observation_space.low[4], env.observation_space.high[4], SEGMENTS)
    th2_av = np.linspace(env.observation_space.low[5], env.observation_space.high[5], SEGMENTS)

    if is_training:
        q = np.zeros(
            shape=(
                len(th1_cos) + 1,
                len(th1_sin) + 1,
                len(th2_cos) + 1,
                len(th2_sin) + 1,
                len(th1_av) + 1,
                len(th2_av) + 1,
                env.action_space.n,
            )
        )
    else:
        q = np.load(Q_NPY_FILE_NAME)

    best_reward = -99999
    rewards_per_episode = []
    i = 0

    while True:
        state = env.reset()[0]

        s_i0 = np.digitize(state[0], th1_cos)
        s_i1 = np.digitize(state[1], th1_sin)
        s_i2 = np.digitize(state[2], th2_cos)
        s_i3 = np.digitize(state[3], th2_sin)
        s_i4 = np.digitize(state[4], th1_av)
        s_i5 = np.digitize(state[5], th2_av)

        terminated = False
        rewards = 0

        while not terminated:
            if is_training and np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(q[s_i0, s_i1, s_i2, s_i3, s_i4, s_i5, :])

            # Take action
            new_state, reward, terminated, _, _ = env.step(action)

            # Convert continuous state
            ns_i0 = np.digitize(new_state[0], th1_cos)
            ns_i1 = np.digitize(new_state[1], th1_sin)
            ns_i2 = np.digitize(new_state[2], th2_cos)
            ns_i3 = np.digitize(new_state[3], th2_sin)
            ns_i4 = np.digitize(new_state[4], th1_av)
            ns_i5 = np.digitize(new_state[5], th2_av)

            # Update Q table
            if is_training:
                q[s_i0, s_i1, s_i2, s_i3, s_i4, s_i5, action] += learning_rate_a * (
                    reward
                    + discount_factor_g * np.max(q[ns_i0, ns_i1, ns_i2, ns_i3, ns_i4, ns_i5, :])
                    - q[s_i0, s_i1, s_i2, s_i3, s_i4, s_i5, action]
                )

            # Set state to new state
            s_i0 = ns_i0
            s_i1 = ns_i1
            s_i2 = ns_i2
            s_i3 = ns_i3
            s_i4 = ns_i4
            s_i5 = ns_i5

            # Collect rewards
            rewards += reward

        if rewards > best_reward:
            best_reward = rewards
            if is_training:
                np.save(Q_NPY_FILE_NAME, q)

        rewards_per_episode.append(rewards)

        # Print stats
        if is_training and i != 0 and i % 100 == 0:
            # Calculate mean reward
            mean_reward = np.mean(rewards_per_episode[len(rewards_per_episode) - 100 :])
            print(f"Episode: {i}, Epsilon: {epsilon:0.2f}, Best Reward: {best_reward:0.1f}, Mean Rewards {mean_reward:0.1f}")
        elif not is_training:
            print(f"Episode: {i}, Rewards: {rewards:0.1f}")

        # Stop if solved
        if best_reward > env.spec.reward_threshold:
            break

        epsilon = max(epsilon - epsilon_decay_rate, epsilon_min)

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


def main():
    np.random.seed(SEED)

    is_training: bool = "true" == os.getenv("IS_TRAINING")
    render: bool = "true" == os.getenv("RENDER")

    run(is_training=is_training, render=render)


if __name__ == "__main__":
    main()
