import gymnasium as gym

from part2_agent.taxi import choose_action, update_q



env = gym.make(
    "FrozenLake-v1",
    map_name="8x8",
    is_slippery=True,
    render_mode="ansi"
)

state, _ = env.reset()

EPISODES = 2000
ALPHA = 0.1
GAMMA = 0.95

EPSILON = 1.0
EPSILON_MIN = 0.05
EPSILON_DECAY = 0.9995

Q = [[0] * env.action_space.n for _ in range(env.observation_space.n)]

success_history = []

for episode in range(EPISODES):
    state, _ = env.reset()
    done = False

    while not done:
        # Choose an action with epsilon-greedy
        action = choose_action(Q, state, EPSILON, env)
        # Tacke the action
        next_state, reward, terminated, truncated, _ = env.step(action)
        # Update Q using the default reward
        update_q(Q,state,action,reward,next_state,terminated,ALPHA,GAMMA)
        # Update state and done
        state = next_state
        done = truncated or terminated
    # Record 1 for success, 0 for failure
    success = 1 if terminated and reward == 1 else 0
    success_history.append(success)

    EPSILON = max(EPSILON_MIN, EPSILON * EPSILON_DECAY)

success_rate = sum(success_history) / len(success_history) * 100

print(f"\nTraining episodes: {EPISODES}")
print(f"Successful episodes: {sum(success_history)}")
print(f"Success rate: {success_rate:.2f}%")


print(env.render())

env.close()