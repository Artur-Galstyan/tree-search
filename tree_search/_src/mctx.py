import equinox as eqx
import jax
import jax.numpy as jnp
from beartype.typing import Any, Callable, NamedTuple
from jaxtyping import Array, Bool, Float, Int, PyTree


NO_PARENT = -1
UNVISITED = -1
ROOT_INDEX = 0


def apply_updates(tree: PyTree, updates) -> PyTree:
    for accessor, update in updates:
        tree = eqx.tree_at(accessor, tree, update)
    return tree


class Tree(eqx.Module):
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


class StepFnInput(NamedTuple):
    embedding: Any
    action: Int[Array, ""]


class StepFnReturn(NamedTuple):
    value: Float[Array, ""]
    discount: Float[Array, ""]
    reward: Float[Array, ""]
    embedding: Any


class ExpansionOutput(NamedTuple):
    node_index: Int[Array, ""]
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


def expansion(
    tree: Tree,
    parent_index: Int[Array, ""],
    action: Int[Array, ""],
    next_node_index: Int[Array, ""],
    step_fn: Callable[[StepFnInput], StepFnReturn],
) -> ExpansionOutput:
    embedding = tree.embeddings[parent_index]
    value, discount, reward, next_state = step_fn(
        StepFnInput(embedding=embedding, action=action)
    )

    new_children_indices = tree.children_indices.at[parent_index, action].set(
        next_node_index
    )
    new_action_from_parent = tree.action_from_parent.at[next_node_index].set(action)
    new_parent_indices = tree.parent_indices.at[next_node_index].set(parent_index)
    new_node_values = tree.node_values.at[next_node_index].set(value)
    new_node_visits = tree.node_visits.at[next_node_index].set(1)
    new_node_discounts = tree.children_discounts.at[parent_index, action].set(discount)
    new_node_rewards = tree.children_rewards.at[parent_index, action].set(reward)
    new_node_embeddings = tree.embeddings.at[next_node_index].set(next_state.embedding)

    updates = [
        (lambda t: t.children_indices),
        new_children_indices,
        (lambda t: t.action_from_parent),
        new_action_from_parent,
        (lambda t: t.parent_indices),
        new_parent_indices,
        (lambda t: t.node_values),
        new_node_values,
        (lambda t: t.node_visits),
        new_node_visits,
        (lambda t: t.children_discounts),
        new_node_discounts,
        (lambda t: t.children_rewards),
        new_node_rewards,
        (lambda t: t.embeddings),
        new_node_embeddings,
    ]

    tree = apply_updates(tree, updates)

    return ExpansionOutput(
        node_index=next_node_index,
        action=action,
    )


def backpropagate(tree: Tree, leaf_index: Int[Array, ""]) -> Tree:
    class BackpropagationState(NamedTuple):
        tree: Tree
        idx: Int[Array, ""]
        value: Float[Array, ""]

    def _backpropagate(state: BackpropagationState) -> BackpropagationState:
        tree, idx, value = state
        parent = tree.parent_indices[idx]
        action = tree.action_from_parent[idx]

        reward = tree.children_rewards[parent, action]
        discount = tree.children_discounts[parent, action]

        parent_value = tree.node_values[parent]
        parent_visits = tree.node_visits[parent]

        leaf_value = reward + discount * state.value
        parent_value = (parent_value * parent_visits + leaf_value) / (
            parent_visits + 1.0
        )

        updates = [
            (lambda t: t.node_values, tree.node_values.at[parent].set(parent_value)),
            (
                lambda t: t.node_visits,
                tree.node_visits.at[parent].set(parent_visits + 1),
            ),
            (
                lambda t: t.children_values,
                tree.children_values.at[parent, action].set(tree.node_values[idx]),
            ),
            (
                lambda t: t.children_visits,
                tree.children_visits.at[parent, action].set(
                    tree.children_visits[parent, action] + 1
                ),
            ),
        ]

        tree = apply_updates(tree, updates)
        return BackpropagationState(idx=parent, value=leaf_value, tree=tree)

    state = BackpropagationState(
        idx=leaf_index, value=tree.node_values[leaf_index], tree=tree
    )

    state = jax.lax.while_loop(
        lambda state: state.idx != ROOT_INDEX, _backpropagate, state
    )
    return state.tree
