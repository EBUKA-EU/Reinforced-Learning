
from q_learning import Q
from environment import GOAL

def get_best_path(q_grid, start_state, goal):
    """
    Find the greedy path from a starting room to the goal
    using a trained Q-table.

    Args:
        q_grid (list[list[float]]): Trained Q-table.
        start_state (int): Starting room.
        goal (int): Goal state.

    Returns:
        list[int]: Sequence of states visited, including
                   the starting room and goal.
    """
    path = [start_state]
    state = start_state

    # Continue until the agent reaches the goal.
    while state != goal:
        # Find the action with the highest Q-value
        # in the current state's row.
        best_action = q_grid[state].index(max(q_grid[state]))

        # Move to the next state.
        next_state = best_action

        # Add the next state to the path.
        path.append(next_state)

        # Update state
        state = next_state

    # Return the completed path.
    return path
