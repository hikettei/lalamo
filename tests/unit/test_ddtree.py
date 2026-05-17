import jax.numpy as jnp

from lalamo.speculator.proposers.ddtree import ddtree_nodes_from_logits


def test_ddtree_nodes_follow_best_first_prefix_scores() -> None:
    logits = jnp.log(
        jnp.asarray(
            [[[0.6, 0.3, 0.1], [0.5, 0.25, 0.25]]],
            dtype=jnp.float32,
        ),
    )

    token_ids, parent_indices, depths, node_mask = ddtree_nodes_from_logits(logits, 4)

    assert jnp.array_equal(token_ids, jnp.asarray([[0, 1, 0, 1]], dtype=jnp.int32))
    assert jnp.array_equal(parent_indices, jnp.asarray([[0, 0, 1, 1]], dtype=jnp.int32))
    assert jnp.array_equal(depths, jnp.asarray([[1, 1, 2, 2]], dtype=jnp.int32))
    assert jnp.array_equal(node_mask, jnp.asarray([[True, True, True, True]]))
