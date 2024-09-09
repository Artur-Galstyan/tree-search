import functools

import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.patches import FancyArrowPatch
from tree_search import (
    Action,
    backpropagate,
    expansion,
    Node,
    selection,
    StepFnInput,
    StepFnReturn,
)
from tree_search._src.mcts import ActionSelectionInput, ActionSelectionReturn


def fixed_layout(G):
    layout = {
        0: (1.5, 2),  # Root node
        1: (0.5, 1),  # Left child of root
        2: (2.5, 1),  # Right child of root
        3: (1.75, 0),  # Left child of left child of root
        4: (1, 0),  # Right child of left child of root
        5: (2.75, 0),  # Left child of right child of root
        6: (0, 0),  # Right child of right child of root
    }
    return {node: layout[node.index] for node in G.nodes()}


def render(root_node: Node, message) -> None:
    graph = nx.MultiDiGraph()

    def _render(node: Node) -> None:
        for action, child in node.child_nodes.items():
            graph.add_node(child)
            graph.add_edge(node, child, action=action)
            _render(child)

    graph.add_node(root_node)
    _render(root_node)

    pos = fixed_layout(graph)

    nx.draw(
        graph,
        pos=pos,
        with_labels=False,  # Changed to False to use custom labels
        node_size=5000,
        linewidths=0.5,
        node_color="none",
        edgecolors="black",
        arrows=True,
        connectionstyle="arc3,rad=0.1",  # Slight curve to reduce overlap
    )
    # Add custom labels
    labels = {
        node: f"I: {node.index}\nP: {node.parent.index if node.parent else 'None'}\nVI: {node.value:.2f}\nVs: {node.visits}"
        for node in graph.nodes()
    }

    nx.draw_networkx_labels(graph, pos, labels, font_size=8)
    plt.axis("off")

    plt.text(1.2, 0.90, message, fontsize=12, fontweight="bold", color="black")
    # set start zoom to be zoomed out
    plt.xlim(0, 3)
    plt.ylim(-1, 3)

    plt.show()


def counter():
    idx = 1

    def increment():
        nonlocal idx
        idx += 1
        return idx - 1

    return increment


ROOT_INDEX = 0


class BanditEnvironment:
    """
        This game tree looks like this:

            0
        / \\
        1   2
        / \\ / \\
        3   4 5  6
    """

    def __init__(self):
        self.tree = {0: [1, 2], 1: [3, 4], 2: [5, 6], 3: [], 4: [], 5: [], 6: []}
        self.current_state = np.array(0)

    def reset(self):
        self.current_state = np.array(0)
        return self.current_state

    def set_state(self, state):
        assert state in [0, 1, 2, 3, 4, 5, 6]
        self.current_state = state

    def step(self, action):
        if self.current_state in [3, 4, 5, 6]:
            return self.current_state, 0, True

        if action < 0 or action >= len(self.tree[int(self.current_state)]):
            raise ValueError("Invalid action")

        self.current_state = self.tree[int(self.current_state)][action]

        done = self.current_state in [3, 4, 5, 6]
        reward = 1 if self.current_state == 6 else 0

        return self.current_state, reward, done

    def render(self):
        print(f"Current state: {self.current_state}")

    @staticmethod
    def get_future_value(state):
        if state == 2:
            return 0.5
        elif state == 6:
            return 1
        else:
            return 0


def ucb1(avg_node_value, visits_parent, visits_node, exploration_exploitation_factor=2):
    return avg_node_value + exploration_exploitation_factor * np.sqrt(
        np.log(visits_parent) / visits_node
    )


def get_root_node(env: BanditEnvironment) -> Node:
    obs = env.reset()
    return Node(parent=None, index=ROOT_INDEX, embedding=obs)


def inner_simulation_fn(
    action_selection_input: ActionSelectionInput, n_actions: int
) -> ActionSelectionReturn:
    node, action = action_selection_input
    best_action = -1
    best_ucb = float("-inf")
    for action in range(n_actions):
        if not node.is_child_visited(Action(action)):
            return ActionSelectionReturn(action=Action(action))
        else:
            child = node.child_nodes[Action(action)]
            ucb = ucb1(
                avg_node_value=child.value,
                visits_parent=node.visits,
                visits_node=child.visits,
            )
            if ucb > best_ucb:
                best_ucb = ucb
                best_action = action
    return ActionSelectionReturn(action=Action(best_action))


def stepper(step_fn_input: StepFnInput, env: BanditEnvironment) -> StepFnReturn:
    env.set_state(step_fn_input.embedding)
    discount = 1.0
    next_state, reward, done = env.step(step_fn_input.action.action)
    value = env.get_future_value(next_state)
    return StepFnReturn(
        value=value, embedding=np.array(next_state), discount=discount, reward=reward
    )


def main():
    env = BanditEnvironment()
    root_node = get_root_node(env)

    max_depth = 2
    n_actions = 2

    n_iterations = 1000

    inner_simulation_fn_partial = functools.partial(
        inner_simulation_fn, n_actions=n_actions
    )

    step_fn_partial = functools.partial(stepper, env=env)
    increment = counter()
    for i in range(n_iterations):
        sim_out = selection(
            root_node=root_node,
            max_depth=max_depth,
            action_selection_fn=inner_simulation_fn_partial,
        )

        leaf_node = expansion(
            sim_out.node_to_expand,
            sim_out.action_to_use,
            step_fn=step_fn_partial,
            next_node_index=increment()
            if sim_out.action_to_use not in sim_out.node_to_expand.child_nodes
            else 0,
        )

        backpropagate(leaf_node)

    node = root_node
    env.reset()
    print(f"{node=}")
    while True:
        best_action = max(node.child_nodes, key=lambda x: node.child_nodes[x].value)
        print(f"Best action: {best_action}")
        obs, reward, done = env.step(best_action.action)
        print(f"Observation: {obs}, Reward: {reward}")
        node = node.child_nodes[best_action]
        print(f"{node=}")
        if done:
            break


if __name__ == "__main__":
    main()
