import heapq
import pandas as pd

class Node:
    """Class representing a node in the Branch and Bound tree"""
    def __init__(self, level, value, weight, bound, items_selected):
        self.level = level
        self.value = value
        self.weight = weight
        self.bound = bound
        self.items_selected = items_selected

    def __lt__(self, other):
        # Reversing comparison to create a max heap
        return self.bound > other.bound


def bound(node, n, W, revenues, times):
    """Calculate upper bound on maximum revenue that can be obtained"""
    if node.weight >= W:
        return 0  # Infeasible solution
    upper_bound = node.value
    j = node.level + 1
    total_weight = node.weight

    # Calculate the upper bound using fractional knapsack logic
    while j < n and total_weight + times[j] <= W:
        total_weight += times[j]
        upper_bound += revenues[j]
        j += 1

    # Add fractional part if more weight can be taken
    if j < n:
        upper_bound += (W - total_weight) * revenues[j] / times[j]

    return upper_bound


def branch_and_bound(revenues, times, W):
    """Branch and Bound algorithm to solve the 0/1 knapsack problem"""
    n = len(revenues)
    queue = []
    root = Node(level=-1, value=0, weight=0, bound=0, items_selected=[])
    root.bound = bound(root, n, W, revenues, times)
    heapq.heappush(queue, root)

    max_value = 0
    best_items = []

    while queue:
        node = heapq.heappop(queue)

        # If node is promising
        if node.bound > max_value:
            # Left child (include next item)
            level = node.level + 1
            if level < n:
                left_child = Node(
                    level=level,
                    value=node.value + revenues[level],
                    weight=node.weight + times[level],
                    bound=0,
                    items_selected=node.items_selected + [1]
                )
                if left_child.weight <= W and left_child.value > max_value:
                    max_value = left_child.value
                    best_items = left_child.items_selected

                left_child.bound = bound(left_child, n, W, revenues, times)
                if left_child.bound > max_value:
                    heapq.heappush(queue, left_child)

            # Right child (exclude next item)
            right_child = Node(
                level=level,
                value=node.value,
                weight=node.weight,
                bound=0,
                items_selected=node.items_selected + [0]
            )
            right_child.bound = bound(right_child, n, W, revenues, times)
            if right_child.bound > max_value:
                heapq.heappush(queue, right_child)

    return max_value, best_items


def solve_knapsack_with_input(df, total_time):
    """Solve the knapsack problem with an input dataframe"""
    revenues = df['Revenue'].tolist()
    times = df['Days'].tolist()
    max_revenue, best_items = branch_and_bound(revenues, times, total_time)
    solution = {'Optimal Revenue': max_revenue, 'Projects Selected': best_items}
    return solution


# Example usage
if __name__ == "__main__":
    # Input table
    data = {
        'Project': [1, 2, 3, 4, 5, 6],
        'Revenue': [15, 20, 5, 25, 22, 17],
        'Days': [51, 60, 35, 60, 53, 10]
    }
    total_time = 100  # Available researcher days
    df = pd.DataFrame(data)

    # Solve the problem
    solution = solve_knapsack_with_input(df, total_time)

    # Print results
    print("Optimal Revenue:", solution['Optimal Revenue'])
    print("Projects Selected (0=No, 1=Yes):", solution['Projects Selected'])
    for i, selected in enumerate(solution['Projects Selected']):
        print(f"Project {i+1}: {'Taken' if selected else 'Not Taken'}")
