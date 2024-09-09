import jax
import jax.numpy as jnp
from beartype.typing import Any, Callable, NamedTuple
from jaxtyping import Array, Bool, Float, Int, PyTree


NO_PARENT = -1
UNVISITED = -1
ROOT_INDEX = 0


class Tree(NamedTuple):
    # Tree Navigation
    parent_indices: Int[Array, "n_nodes"]
    children_indices: Int[Array, "n_nodes n_actions"]
    action_from_parent: Int[Array, "n_nodes"]

    raw_values: Float[Array, "n_nodes"]

    node_visits: Int[Array, "n_nodes"]
    node_values: Float[Array, "n_nodes"]

    children_values: Float[Array, "n_nodes n_actions"]
    children_visits: Int[Array, "n_nodes n_actions"]
    children_rewards: Float[Array, "n_nodes n_actions"]
    children_discounts: Float[Array, "n_nodes n_actions"]
    children_prior_logits: Float[Array, "n_nodes n_actions"]

    embeddings: PyTree


class RootFnOutput(NamedTuple):
    embedding: Any


class ActionSelectionInput(NamedTuple):
    tree: Tree
    node_index: Int[Array, ""]
    depth: Int[Array, ""]


class ActionSelectionReturn(NamedTuple):
    action: Int[Array, ""]


class SelectionOutput(NamedTuple):
    parent_index: Int[Array, ""]
    action: Int[Array, ""]


def generate_tree(n_nodes: int, n_actions: int, root_fn_output: RootFnOutput) -> Tree:
    parent_indices = jnp.full(shape=(n_nodes), fill_value=NO_PARENT)
    action_from_parent = jnp.full(shape=(n_nodes), fill_value=NO_PARENT)
    children_indices = jnp.full(shape=(n_nodes, n_actions), fill_value=UNVISITED)

    raw_values = jnp.zeros(shape=(n_nodes))

    node_visits = jnp.zeros(shape=(n_nodes), dtype=jnp.int32)
    node_values = jnp.zeros(shape=(n_nodes))

    children_values = jnp.zeros(shape=(n_nodes, n_actions))
    children_visits = jnp.zeros(shape=(n_nodes, n_actions), dtype=jnp.int32)
    children_rewards = jnp.zeros(shape=(n_nodes, n_actions))
    children_discounts = jnp.zeros(shape=(n_nodes, n_actions))
    children_prior_logits = jnp.zeros(shape=(n_nodes, n_actions))

    embeddings = jax.tree.map(
        lambda x: jnp.broadcast_to(x, (n_nodes, *x.shape)), root_fn_output.embedding
    )
    embeddings = embeddings.at[ROOT_INDEX].set(root_fn_output.embedding)

    return Tree(
        parent_indices=parent_indices,
        children_indices=children_indices,
        action_from_parent=action_from_parent,
        raw_values=raw_values,
        node_visits=node_visits,
        node_values=node_values,
        children_values=children_values,
        children_visits=children_visits,
        children_rewards=children_rewards,
        children_discounts=children_discounts,
        children_prior_logits=children_prior_logits,
        embeddings=embeddings,
    )


def selection(
    tree: Tree,
    max_depth: int,
    inner_action_selection_fn: Callable[[ActionSelectionInput], ActionSelectionReturn],
) -> SelectionOutput:
    class SelectionState(NamedTuple):
        node_index: Int[Array, ""]
        next_node_index: Int[Array, ""]
        depth: Int[Array, ""]
        action: Int[Array, ""]
        proceed: Bool[Array, ""]

    def _selection(state: SelectionState) -> SelectionState:
        node_index = state.next_node_index
        action_selection_output = inner_action_selection_fn(
            ActionSelectionInput(tree, node_index, state.depth)
        )
        child_index = tree.children_indices[node_index, action_selection_output.action]
        proceed = child_index != UNVISITED and state.depth + 1 < max_depth

        return SelectionState(
            node_index=node_index,
            next_node_index=child_index,
            depth=state.depth + 1,
            action=action_selection_output.action,
            proceed=proceed,
        )

    state = SelectionState(
        node_index=jnp.array(NO_PARENT),
        next_node_index=jnp.array(ROOT_INDEX),
        depth=jnp.array(0),
        action=jnp.array(UNVISITED),
        proceed=jnp.array(True),
    )

    final_state = jax.lax.while_loop(lambda state: state.proceed, _selection, state)

    return SelectionOutput(
        parent_index=final_state.node_index,
        action=final_state.action,
    )
