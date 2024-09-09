import jax
import jax.numpy as jnp
from tree_search._src.mctx import RootFnOutput


n_nodes = 7

root_fn_output = RootFnOutput(embedding=jnp.zeros((3, 4)))

embeddings = jax.tree.map(
    lambda x: jnp.broadcast_to(x, (n_nodes, *x.shape)), root_fn_output.embedding
)

print(embeddings, embeddings.shape)
