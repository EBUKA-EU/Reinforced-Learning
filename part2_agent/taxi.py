
import gymnasium as gym

env = gym.make("Taxi-v4", render_mode="ansi")

# Start a new episode
state, info = env.reset()

print(f"Initial state: {state}")
print(f"Number of states: {env.observation_space.n}")
print(f"Number of actions: {env.action_space.n}")

print(env.render())

# Select a random action using Gymnasium
action = env.action_space.sample()

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