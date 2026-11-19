
import random
import gymnasium as gym

env = gym.make("Taxi-v4", render_mode="ansi")

# Initialize a Q-table with 500 states and 6 actions
Q = [[0] * env.action_space.n for _ in range(env.observation_space.n)]

def choose_action(q_grid, state, epsilon, env):
    """
    Choose an action using epsilon-greedy exploration.

    Args:
        q_grid (list[list[float]]): Q-table.
        state (int): Current Taxi state.
        epsilon (float): Exploration probability.
        env: Gymnasium Taxi environment.

    Returns:
        int: Selected action.
    """
    # Generate a random float between 0 and 1
    random_value = random.random()

    if random_value < epsilon:
        return env.action_space.sample()
    else:
        return q_grid[state].index(max(q_grid[state]))

def update_q(q_grid, state, action, reward, next_state, terminated, alpha, gamma):
    """
    Update one Q-value using the Q-learning rule.

    Args:
        q_grid (list[list[float]]): Q-table.
        state (int): Current state.
        action (int): Action taken.
        reward (float): Immediate reward.
        next_state (int): State after the action.
        terminated (bool): Whether a terminal state was reached.
        alpha (float): Learning rate.
        gamma (float): Discount factor.
    """
    # Retrieve the current Q-value
    old_q_value = q_grid[state][action]

    # Calculate the maximum future Q-value, zero if terminated
    if terminated:
        max_future_q = 0
    else:
        max_future_q = max(q_grid[next_state])

    # Calculate the target
    target = reward + gamma * max_future_q

    # Calculate the updated Q
    new_q_value = old_q_value + alpha * (target - old_q_value)

    q_grid[state][action] = new_q_value


if __name__ == "__main__":

    # Start a new episode
    state, info = env.reset()

    print(f"Initial state: {state}")
    print(f"Number of states: {env.observation_space.n}")
    print(f"Number of actions: {env.action_space.n}")

    print(env.render())

    # Select a random action using Gymnasium
    action = choose_action(Q, state, 1.0, env)

    # Execute the action and unpack all five returned values
    next_state, reward, terminated, truncated, info = env.step(action)

    # Determine wheter the episode is ended
    done = truncated or terminated

    print(f"Action: {action}")
    print(f"Next state: {next_state}")
    print(f"Reward: {reward}")
    print(f"Episode finished?: {done}")

    print(env.render())

    env.close()

