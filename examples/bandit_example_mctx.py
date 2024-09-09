import jax.numpy as jnp
import numpy as np


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
