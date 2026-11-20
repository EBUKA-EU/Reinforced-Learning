import gymnasium as gym
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

    env.close()

    return q_grid, success_history


env = gym.make(
    "FrozenLake-v1",
    map_name="8x8",
    is_slippery=True,
    render_mode="ansi"
)

EPISODES = 2000
ALPHA = 0.1
GAMMA = 0.95

EPSILON = 1.0
EPSILON_MIN = 0.05
EPSILON_DECAY = 0.9995

default_q, default_history = train_frozenlake(env,2000,use_modified_reward=False)
modified_q, modified_history = train_frozenlake(env, 2000, use_modified_reward=True)

default_rate = sum(default_history) / len(default_history) * 100
modified_rate = sum(modified_history) / len(modified_history) * 100


print(f"Default reward success rate: {default_rate:.2f}%")
print(f"Modified reward success rate: {modified_rate:.2f}%")

env.close()