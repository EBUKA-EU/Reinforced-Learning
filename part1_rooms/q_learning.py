import random
from environment import NUM_STATES,GOAL,R,WALL_PENALTY


GAMMA = 0.8
EPISODE_COUNT = 600

Q = [[0] * NUM_STATES for _ in range(NUM_STATES)]

for episode in range(EPISODE_COUNT):
    state = random.randrange(NUM_STATES)

    while state != GOAL:
        action = random.randrange(NUM_STATES)

        reward = R[state][action]

        if reward == WALL_PENALTY:
            next_state = state
        else:
            next_state = action

        max_q_value = max(Q[next_state])
        Q[state][action] = R[state][action] + GAMMA * max_q_value

        state = next_state


for row in Q:
    print(row)