import functools

import jax.numpy as jnp
import numpy as np
from jaxtyping import Array
from tree_search._src.mctx import (
    ActionSelectionInput,
    ActionSelectionReturn,
    MCTX,
    RootFnOutput,
    StepFnInput,
    StepFnReturn,
    UNVISITED,
)


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


def get_root_node(env: BanditEnvironment) -> RootFnOutput:
    obs = jnp.array(env.reset())
    return RootFnOutput(embedding=obs)


def inner_simulation_fn(
    action_selection_input: ActionSelectionInput, n_actions: int
) -> ActionSelectionReturn:
    tree, node_index, depth = action_selection_input
    best_action = -1
    best_ucb = float("-inf")
    for action in range(n_actions):
        if (
            tree.children_indices[node_index, action] == UNVISITED
        ):  # TODO: fix this part
            return ActionSelectionReturn(action=action)
        else:
            child = tree.children_indices[node_index, action]
            ucb = ucb1(
                avg_node_value=tree.node_values[child],
                visits_parent=tree.node_visits[node_index],
                visits_node=tree.node_visits[child],
            )
            if ucb > best_ucb:
                best_ucb = ucb
                best_action = action
    return ActionSelectionReturn(action=best_action)


def stepper(step_fn_input: StepFnInput, env: BanditEnvironment) -> StepFnReturn:
    env.set_state(step_fn_input.embedding)
    discount = 1.0
    next_state, reward, done = env.step(step_fn_input.action)
    value = env.get_future_value(next_state)
    return StepFnReturn(
        value=value, embedding=np.array(next_state), discount=discount, reward=reward
    )


def main():
    env = BanditEnvironment()
    max_depth = 2
    n_actions = 2

    n_iterations = 10

    inner_simulation_fn_partial = functools.partial(
        inner_simulation_fn, n_actions=n_actions
    )

    step_fn_partial = functools.partial(stepper, env=env)
    MCTX.search(
        max_depth=max_depth,
        n_actions=n_actions,
        n_iterations=n_iterations,
        root_fn=functools.partial(get_root_node, env=env),
        inner_action_selection_fn=inner_simulation_fn_partial,
        step_fn=step_fn_partial,
    )


if __name__ == "__main__":
    main()
