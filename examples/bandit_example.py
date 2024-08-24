import functools

import numpy as np
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

    n_iterations = 10

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
            next_node_index=increment(),
        )
        print("\n")
        backpropagate(leaf_node)
        print("\n")

    node = root_node
    env.reset()
    # print(f"{node=}")
    # while True:
    #     best_action = max(node.child_nodes, key=lambda x: node.child_nodes[x].value)
    #     print(f"Best action: {best_action}")
    #     obs, reward, done = env.step(best_action.action)
    #     print(f"Observation: {obs}, Reward: {reward}")
    #     node = node.child_nodes[best_action]
    #     print(f"{node=}")
    #     if done:
    #         break


if __name__ == "__main__":
    main()
