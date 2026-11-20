
import random
import gymnasium as gym
import matplotlib.pyplot as plt

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

EPISODES = 10000
ALPHA = 0.1
GAMMA = 0.95

EPSILON = 1.0
EPSILON_MIN = 0.05
EPSILON_DECAY = 0.9995

episode_rewards = []

WINDOW = 100
moving_averages = []

def main():
    epsilon = EPSILON
    for _ in range(EPISODES):
    # Reset the environment
        state, _  = env.reset()

        done = False
        total_reward = 0

        while not done:
            # Choose an action using the epsilon-greedy
            action = choose_action(Q,state,epsilon,env)

            # Execute the action using
            next_state, reward, terminated, truncated, info = env.step(action)

            # Update the Q-table
            update_q(Q, state, action, reward, next_state,terminated,ALPHA,GAMMA)

            # Accummulate reward
            total_reward += reward

            # Update the current state
            state = next_state

            # Check whether the episode has ended
            done = terminated or truncated

        # Save the total reward earned in this episode
        episode_rewards.append(total_reward)

        # Decay epsilon, but dont let it fall below EPSILON_MIN
        epsilon = max(epsilon * EPSILON_DECAY, EPSILON_MIN)

    env.close()


    # Calculate a moving average 
    # using each consecutive group of 100 rewards
    for i in range(WINDOW, len(episode_rewards) + 1):
        # get the previous window reward
        recent_rewards = episode_rewards[i - WINDOW: i]

        # Calculate the average
        average = sum(recent_rewards) / len(recent_rewards)

        moving_averages.append(average)

    print(f"Training episodes: {EPISODES}")
    print(f"Final epsilon: {epsilon:.4f}")
    print(f"First 100 average reward: {sum(episode_rewards[:100]) / 100:.2f}")
    print(f"Last 100 average reward: {sum(episode_rewards[-100:]) / 100:.2f}")

    plt.plot(range(WINDOW, len(episode_rewards) + 1), moving_averages)
    plt.xlabel("Episode")
    plt.ylabel("Average reward")
    plt.title("Taxi Q-Learning Training Performance")
    plt.savefig("training_performance.png", dpi=150)
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()