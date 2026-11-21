import gymnasium as gym
import matplotlib.pyplot as plt
from part2_agent.taxi import choose_action, update_q

def modify_reward(reward, terminated):
    """
    Apply a custom reward scheme to FrozenLake.

    Args:
        reward (float): Original reward from Gymnasium.
        terminated (bool): Whether the episode terminated.

    Returns:
        float: Modified reward for Q-learning.
    """
    # If the agent reached a goal return the original reward(+1)
    if terminated and reward == 1:
        return reward
    # f the agent fell into a hole
    elif terminated:
        return reward - 1
    # Otherwise apply a small step penalty
    else:
        return reward - 0.01
    
def train_frozenlake(env, episodes, use_modified_reward=False):
    """
    Train a FrozenLake Q-learning agent.

    Args:
        env: Gymnasium FrozenLake environment.
        episodes (int): Number of training episodes.
        use_modified_reward (bool): Whether to apply
            the custom reward function.

    Returns:
        tuple: Trained Q-table and success history.
    """
    # Initialize a fresh Q-table
    q_grid = [[0] * env.action_space.n for _ in range(env.observation_space.n)]

    epsilon = EPSILON
    success_history = []

    for _ in range(episodes):
        state, _ = env.reset()
        done = False

        while not done:
            # choose an epsilon greedy action
            action = choose_action(q_grid, state, epsilon, env)
            # excute the acction
            next_state, reward, terminated, truncated, _ = env.step(action)

            # Start with default environment reward
            training_reward = reward
            
            # if use_modified_reward is True, 
            # we replace the reward used for training.
            if use_modified_reward:
                training_reward = modify_reward(reward, terminated)

            # Update Q with the taining reward
            update_q(q_grid,state,action,training_reward,next_state,terminated,ALPHA,GAMMA)

            # Update state and done
            state = next_state
            done = truncated or terminated

        # record suceess using the original reward
        success = 1 if terminated and reward == 1 else 0
        success_history.append(success)

        # decay epsilon
        epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)

    return q_grid, success_history

def moving_average_success(history, window=100):
    """
    Calculate moving-average success rates.

    Args:
        history (list[int]): Episode outcomes
            where 1 means success and 0 means failure.
        window (int): Number of episodes per window.

    Returns:
        list[float]: Moving-average success percentages.
    """
    averages = []

    for i in range(window, len(history) + 1):
        # extract the previous window outocme
        recent = history[i - window :i]
        #calculate the percentage success
        success_rate = sum(recent)/len(recent) * 100
        averages.append(success_rate)

    return averages



env = gym.make(
    "FrozenLake-v1",
    map_name="8x8",
    is_slippery=True,
    render_mode="ansi"
)

EPISODES = 10000
ALPHA = 0.1
GAMMA = 0.95

EPSILON = 1.0
EPSILON_MIN = 0.05
EPSILON_DECAY = 0.9995

default_q, default_history = train_frozenlake(env,EPISODES,use_modified_reward=False)
modified_q, modified_history = train_frozenlake(env, EPISODES, use_modified_reward=True)

default_rate = sum(default_history) / len(default_history) * 100
modified_rate = sum(modified_history) / len(modified_history) * 100

default_avg = moving_average_success(default_history)
modified_avg = moving_average_success(modified_history)

# Episode numbers, starting at episode 100
episodes = range(100, len(default_history) + 1)

# Plot both reward systems on the same graph
plt.plot(episodes, default_avg, label="Default Rewards")
plt.plot(episodes, modified_avg, label="Modified Rewards")

plt.xlabel("Training Episode")
plt.ylabel("Moving Average Success Rate (%)")
plt.title(f"FrozenLake: Default vs Modified Rewards ({EPISODES} Episodes)")

plt.savefig(f"frozenlake_comparison.png", dpi=150)

plt.legend()
plt.grid(True)
plt.tight_layout()

# Display the graph
plt.show()

print(f"\nTraining episodes: {EPISODES}")

print(f"Default reward success rate: {default_rate:.2f}%")
print(f"Modified reward success rate: {modified_rate:.2f}%")

print(f"Default last 100 success rate: {sum(default_history[-100:])}%")
print(f"Modified last 100 success rate: {sum(modified_history[-100:])}%")

env.close()