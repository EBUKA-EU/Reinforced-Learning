
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

