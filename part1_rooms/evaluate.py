from q_learning import Q,WALL_PENALTY
from environment import R,GOAL

from collections import deque

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


def bfs_shortest_path(reward_matrix, start_state, goal):
    """
    Find a shortest path to the goal using BFS.

    Args:
        reward_matrix (list[list[int]]): Reward matrix.
        start_state (int): Starting room.
        goal (int): Destination room.

    Returns:
        list[int]: A shortest path from start to goal.
    """
    queue = deque([[start_state]])
    visited = {start_state}

    while queue:
        # Remove the first path from the queue.
        first_path = queue.popleft()

        # Find the last room in that path.
        last_room = first_path[-1]

        # If that room is the goal,
        # return the path.
        if last_room == goal:
            return first_path

        # Find neighboring rooms connected
        # by doors (rewards -1 or 100).
        neighbor_rooms = [room for room, reward in enumerate(reward_matrix[last_room]) if reward != WALL_PENALTY and room != last_room]

        # For each unvisited neighbor:
        # mark it visited and enqueue a new path.
        for room in neighbor_rooms:
            if room not in visited:
                visited.add(room)
                new_path = first_path + [room]
                queue.append(new_path)

    return []


for start_state in range(GOAL):
    # get the Q-learning path
    q_path = get_best_path(Q,start_state,GOAL)

    # Get the BFS shortest path
    bfs_path = bfs_shortest_path(R,start_state,GOAL)

    # calculate the number of steps in each path
    q_steps = len(q_path) -1
    bfs_steps = len(bfs_path) - 1

    # Check whether the number of steps matches
    is_shortest = True if len(q_path) == len(bfs_path) else False

    # print both paths
    print(f"Room {start_state}")
    print(f"Q-learing: {q_path}")
    print(f"BFS:       {bfs_path}")
    print(f"Steps: {q_steps}")
    print(f"Shortest?: {is_shortest}")
    print()