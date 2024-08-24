import numpy as np
from beartype.typing import Any, Callable, NamedTuple


ROOT_INDEX = 0


class Action(NamedTuple):
    action: int


class Node:
    index: int

    child_nodes: dict[Action, "Node"]
    parent_node: "Node | None"

    visits: int
    value: float
    discount: float
    reward: float

    embedding: Any

    def __init__(self, parent: "Node | None", index: int, embedding: Any) -> None:
        self.parent = parent
        self.index = index
        self.embedding = embedding

        self.child_nodes = dict()
        self.visits, self.value, self.reward, self.discount = 0, 0, 0, 0

    def is_child_visited(self, action: Action) -> bool:
        return action in self.child_nodes

    def __repr__(self) -> str:
        return (
            f"[Index: {self.index} "
            f"Parent: {self.parent.index if self.parent is not None else None} "
            f"Value: {np.round(self.value, 2)}, Visits: {self.visits}]"
        )


class StepFnReturn(NamedTuple):
    value: float
    discount: float
    reward: float
    embedding: Any


class StepFnInput(NamedTuple):
    embedding: Any
    action: Action


class SelectionOutput(NamedTuple):
    node_to_expand: Node
    action_to_use: Action


class ActionSelectionInput(NamedTuple):
    node: Node
    depth: int


class ActionSelectionReturn(NamedTuple):
    action: Action


def selection(
    root_node: Node,
    max_depth: int,
    action_selection_fn: Callable[[ActionSelectionInput], ActionSelectionReturn],
) -> SelectionOutput:
    class SelectionState(NamedTuple):
        node: Node
        next_node: Node | None
        action: Action
        depth: int
        proceed: bool

    def _select(state: SelectionState) -> SelectionState:
        node = state.next_node
        assert node is not None
        action_selection_output = action_selection_fn(
            ActionSelectionInput(node, state.depth)
        )
        child_visited = node.is_child_visited(action_selection_output.action)
        if not child_visited:
            next_node = None
        else:
            next_node = node.child_nodes[action_selection_output.action]
        proceed = child_visited and state.depth + 1 < max_depth

        return SelectionState(
            node=node,
            next_node=next_node,
            action=action_selection_output.action,
            depth=state.depth + 1,
            proceed=proceed,
        )

    state = SelectionState(
        node=root_node, next_node=root_node, action=Action(0), depth=0, proceed=True
    )

    while state.proceed:
        state = _select(state)

    return SelectionOutput(node_to_expand=state.node, action_to_use=state.action)


def expansion(
    node: Node,
    action: Action,
    next_node_index: int,
    step_fn: Callable[[StepFnInput], StepFnReturn],
) -> Node:
    value, discount, reward, next_state = step_fn(
        StepFnInput(embedding=node.embedding, action=action)
    )

    if action in node.child_nodes:
        new_node = node.child_nodes[action]
    else:
        new_node = Node(parent=node, index=next_node_index, embedding=next_state)
        node.child_nodes[action] = new_node

    new_node.value = value
    new_node.reward = reward
    new_node.discount = discount
    new_node.visits += 1

    return new_node


def backpropagate(leaf_node: Node) -> None:
    class BackpropagationState(NamedTuple):
        node: Node
        value: float

    def _backpropagate(state: BackpropagationState) -> BackpropagationState:
        print(state)
        parent = state.node.parent
        assert parent is not None

        leaf_value = state.node.reward + state.node.discount * state.value
        print(f"Leaf value: {leaf_value}")
        parent_value = (parent.value * parent.visits + leaf_value) / (
            parent.visits + 1.0
        )
        print(f"Parent value: {parent_value}")

        parent.value = parent_value
        parent.visits += 1
        print(f"Parent visits: {parent.visits}")

        return BackpropagationState(node=parent, value=leaf_value)

    state = BackpropagationState(node=leaf_node, value=leaf_node.value)

    while state.node.parent is not None:
        state = _backpropagate(state)


class MCTS:
    @staticmethod
    def search(
        max_depth: int,
        n_actions: int,
        root_node_fn: Callable[[], Node],
        action_selection_fn: Callable[[ActionSelectionInput], ActionSelectionReturn],
        step_fn: Callable[[StepFnInput], StepFnReturn],
        n_iterations: int,
    ):
        root_node = root_node_fn()
        node_index_counter = 0
        for _ in range(n_iterations):
            selection_output = selection(
                root_node=root_node,
                max_depth=max_depth,
                action_selection_fn=action_selection_fn,
            )

            if (
                selection_output.action_to_use
                not in selection_output.node_to_expand.child_nodes
            ):
                node_index_counter += 1

            leaf_node = expansion(
                node=selection_output.node_to_expand,
                action=selection_output.action_to_use,
                next_node_index=node_index_counter,
                step_fn=step_fn,
            )
            backpropagate(leaf_node)
