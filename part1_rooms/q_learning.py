import random
from environment import NUM_STATES,GOAL,R,WALL_PENALTY


GAMMA = 0.8
EPISODE_COUNT = 10000

def train_agent():
    """
    Train a Q-learning agent in the eight-state rooms environment.

    Returns:
        list[list[float]]: The trained Q-table.
    """

    Q = [[0] * NUM_STATES for _ in range(NUM_STATES)]

    for _ in range(EPISODE_COUNT):
        state = random.randrange(NUM_STATES)

        while state != GOAL:
            action = random.randrange(NUM_STATES)

            reward = R[state][action]

            if reward == WALL_PENALTY:
                next_state = state
            else:
                next_state = action

            if next_state == GOAL:
                max_q_value = 0
            else:
                max_q_value = max(Q[next_state])

            Q[state][action] = R[state][action] + GAMMA * max_q_value

            state = next_state

    return Q

if __name__ == "__main__":
    Q = train_agent()

    for state, row in enumerate(Q):
        print(f"{state}: {[round(value, 2) for value in row]}")
