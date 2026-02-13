import os
import random
from collections import deque
from typing import Any, List, Tuple

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

SEED = 1770648564
POLICY_FILE_NAME = r"frozen_lake_deep_q_learning_policy.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed=SEED):
    """Sets the seed for reproducibility across Python, NumPy, and PyTorch."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # for multi-GPU setups
        # For deterministic behavior on GPU (might impact performance)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    if torch.mps.is_available():
        torch.mps.manual_seed(seed)
    print(f"Random seed set to {seed}")


# Define model
class DQN(nn.Module):
    def __init__(self, in_states, h1_nodes, out_actions):
        super(DQN, self).__init__()

        self.num_states = in_states

        # Define network layers
        self.model = nn.Sequential(
            nn.Linear(in_states, h1_nodes),
            nn.ReLU(),
            nn.Linear(h1_nodes, out_actions),
        )

    def forward(self, x):
        return self.model(x)


class ReplayMemory:
    def __init__(self, max_length):
        self.memory = deque([], maxlen=max_length)

    def append(self, transition):
        self.memory.append(transition)

    def sample(self, sample_size) -> List[Tuple[Any, Any, Any, int, bool]]:
        return random.sample(self.memory, sample_size)

    def __len__(self):
        return len(self.memory)


class FrozenLakeDQL:
    # Hyperparameters (adjustable)
    learning_rate_a = 0.001  # learning rate (alpha)
    discount_factor_g = 0.9  # discount rate (gamma)
    network_sync_rate = 100  # number of steps the agent takes before syncing the weight and bias from policy -> target network
    replay_memory_size = 1000
    mini_batch_size = 32  # size of the training data set sampled from the replay memory

    # Neural network
    loss_fn: nn.MSELoss = nn.MSELoss()  # NN loss function. MSE=Mean Squared Error. Can be swapped to something else.
    optimizer = None  # NN optimizer. Initialize later.

    ACTIONS = ["L", "D", "R", "U"]  # for printing 0,1,2,3 => (L)eft, (D)own, (R)ight, (U)p

    def train(self, episodes: int, render: bool = False, is_slippery: bool = False):
        env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=is_slippery, render_mode="human" if render else None)
        env.observation_space.seed(SEED)
        env.action_space.seed(SEED)
        num_states = env.observation_space.n
        num_actions = env.action_space.n

        epsilon = 1
        memory = ReplayMemory(self.replay_memory_size)

        # Create policy and target networks. Number of nodes in hidden layer can be adjusted.
        policy_dqn = DQN(in_states=num_states, h1_nodes=num_states, out_actions=num_actions).to(DEVICE)
        target_dqn = DQN(in_states=num_states, h1_nodes=num_states, out_actions=num_actions).to(DEVICE)

        # Copy everything from policy -> target DQN.
        target_dqn.load_state_dict(policy_dqn.state_dict(), strict=True)

        print("Policy (random, before training):")
        self.print_dqn(policy_dqn)

        # Policy network optimizer. "Adam" optimizer can be swapped with something else
        self.optimizer = torch.optim.Adam(policy_dqn.parameters(), lr=self.learning_rate_a)

        # List to keep track of rewards collected per episode. Initialize list to 0s.
        rewards_per_episode = np.zeros(episodes)

        # List to keep track of epsilon decay
        epsilon_history = []

        # Track number of steps taken. Used for syncing policy => target network
        step_count = 0

        for i in range(episodes):
            state = env.reset(seed=SEED)[0]
            terminated = False  # True when agent falls in hole or reaches goal
            truncated = False  # True when agent takes more than 200 actions

            while not terminated and not truncated:
                if np.random.rand() < epsilon:
                    action = env.action_space.sample()
                else:
                    with torch.no_grad():
                        input_tensor = self.state_to_dqn_input(state, num_states).to(DEVICE)
                        action = policy_dqn(input_tensor).argmax().item()

                # Execute action
                new_state, reward, terminated, truncated, _ = env.step(action)

                # Save experience in replay
                memory.append((state, action, new_state, reward, terminated))

                # Move to the next state
                state = new_state

                step_count += 1

            if reward == 1:
                rewards_per_episode[i] = 1

            # Check if enough experience has been collected and if at least 1 reward has been collected
            if len(memory) > self.mini_batch_size and np.sum(rewards_per_episode) > 0:
                mini_batch: List[Tuple[Any, Any, Any, int, bool]] = memory.sample(self.mini_batch_size)
                self.optimize(mini_batch, policy_dqn, target_dqn)

                # Decay epsilon
                epsilon = max(epsilon - 1 / episodes, 0)
                epsilon_history.append(epsilon)

                # Copy policy network to target network if step_count > network_sync_rate
                if step_count > self.network_sync_rate:
                    target_dqn.load_state_dict(policy_dqn.state_dict(), strict=True)
                    step_count = 0

        # Close environment
        env.close()

        # Save policy
        torch.save(policy_dqn.state_dict(), POLICY_FILE_NAME)

        # Create new graph
        plt.figure(1)

        # Plot average rewards (Y-axis) vs episodes (X-axis)
        sum_rewards = np.zeros(episodes)
        for x in range(episodes):
            sum_rewards[x] = np.sum(rewards_per_episode[max(0, x - 100) : (x + 1)])
        plt.subplot(1, 2, 1)
        plt.plot(sum_rewards)

        # Plot epsilon decay
        plt.subplot(1, 2, 2)
        plt.plot(epsilon_history)

        # Save plot
        plt.savefig("frozen_lake_deep_q_learning.png")

    def optimize(self, mini_batch: List[Tuple[Any, Any, Any, int, bool]], policy_dqn: DQN, target_dqn: DQN):
        # Get number of input nodes
        num_states = policy_dqn.num_states

        current_q_list = []
        target_q_list = []

        for state, action, new_state, reward, terminated in mini_batch:
            if terminated:
                # Agent either reached goal (reward = 1) or fell into hole (reward = 0).
                # When in a terminated state, target q value should be set to the reward.
                target = torch.FloatTensor([reward])
            else:
                # Calculate target q value
                new_state_one_hot_tensor = self.state_to_dqn_input(new_state, num_states).to(DEVICE, copy=True)
                target = torch.FloatTensor(reward + self.discount_factor_g * target_dqn(new_state_one_hot_tensor).detach().cpu()).max()

            # Get the current state of Q values
            current_state_one_hot_tensor = self.state_to_dqn_input(state, num_states)
            current_q = policy_dqn(current_state_one_hot_tensor.to(DEVICE, copy=True))
            current_q_list.append(current_q)

            # Get the target set of Q values
            target_q = target_dqn(current_state_one_hot_tensor.to(DEVICE))
            target_q[action] = target
            target_q_list.append(target_q)

        # Compute loss for the whole mini-batch
        loss = self.loss_fn(torch.stack(current_q_list).to(DEVICE), torch.stack(target_q_list).to(DEVICE))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def state_to_dqn_input(self, state: int, num_states: int) -> torch.Tensor:
        input_tensor = torch.zeros(num_states)
        input_tensor[state] = 1
        return input_tensor

    def print_dqn(self, dqn: DQN):
        # Get number of input nodes
        num_states = dqn.num_states

        for s in range(num_states):
            # Format q values for printing
            q_values = ""
            dqn_input_tensor = self.state_to_dqn_input(s, num_states).to()
            qs = dqn(dqn_input_tensor.to(DEVICE, copy=True)).detach().cpu().tolist()
            # for q in qs:
            #     q_values += "{:+.2f}".format(q) + " "
            # q_values = q_values.rstrip()
            q_values = " ".join(["{:+.2f}".format(q) for q in qs]).rstrip()

            # Map the best action
            best_action = self.ACTIONS[dqn(dqn_input_tensor.to(DEVICE)).detach().cpu().argmax()]

            # Print policy in the format of: state, action, q_values
            # The printed layout matches the FrozenLake map.
            print(f"{s:02}, {best_action}, [{q_values}]", end=" ")
            if (s + 1) % 4 == 0:
                print()

    def test(self, episodes: int, render: bool = False, is_slippery: bool = False):
        env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=is_slippery, render_mode="human" if render else None)
        num_states = env.observation_space.n
        num_actions = env.action_space.n

        # Load learned policy
        policy_dqn = DQN(in_states=num_states, h1_nodes=num_states, out_actions=num_actions).to(DEVICE)
        policy_dqn.load_state_dict(torch.load(POLICY_FILE_NAME))
        policy_dqn.eval()

        print("Policy (trained):")
        self.print_dqn(policy_dqn)

        for i in range(episodes):
            state = env.reset(seed=SEED)[0]  # Initialize to state 0
            terminated = False
            truncated = False

            # Agent navigates map until it falls into a hole (terminated), or has taken 200 steps (truncated)
            while not terminated and not truncated:
                with torch.no_grad():
                    action = policy_dqn(self.state_to_dqn_input(state, num_states).to(DEVICE)).argmax().item()

                # Execute action
                state, reward, terminated, truncated, _ = env.step(action)

        env.close()


def main():
    set_seed(SEED)

    frozen_lake = FrozenLakeDQL()
    is_slippery = False
    frozen_lake.train(1000, is_slippery=is_slippery)
    frozen_lake.test(100, is_slippery=is_slippery)


if __name__ == "__main__":
    main()
