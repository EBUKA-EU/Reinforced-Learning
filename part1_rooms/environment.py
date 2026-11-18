
NUM_STATES = 8
GOAL = 7

STEP_REWARD = -1
WALL_PENALTY = -10
GOAL_REWARD = 100

DOORS = [
    (0, 4),
    (4, 0),
    (1, 3),
    (3, 1),
    (2, 3),
    (3, 2),
    (3, 4),
    (4, 3),
    (1, 5),
    (5, 1),
    (4, 6),
    (6, 4),
    (5, 7),
    (7, 5),
    (6, 7),
    (7, 6)
]

Q = [[0] * NUM_STATES for _ in range(NUM_STATES)]

R = [[WALL_PENALTY] * NUM_STATES for _ in range(NUM_STATES)]
for state, action in DOORS:
    R[state][action] = 100 if action == GOAL else -1

R[GOAL][GOAL] = GOAL_REWARD



