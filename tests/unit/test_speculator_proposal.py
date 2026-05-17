import jax
import jax.numpy as jnp

from lalamo.sampling import SamplingPolicy
from lalamo.speculator.proposal import TrieProposal, fold_gumbel_key


def test_create_root_proposal_returns_root_frontier_with_next_sample_policy() -> None:
    vocabulary_size = 5
    batch_size = 2
    base_policy = (
        SamplingPolicy.init(frequency_penalty=1.0).with_empty_token_counts(vocabulary_size).broadcast(batch_size)
    )
    keys = jax.random.split(jax.random.key(0), batch_size)

    proposal, frontier = TrieProposal.create(
        root_ids=jnp.asarray([2, 3], dtype=jnp.int32),
        root_gumbel_positions=jnp.asarray([7, 9], dtype=jnp.int32),
        sampling_policy=base_policy,
        gumbel_keys=keys,
        vocabulary_size=vocabulary_size,
        budget=4,
    )

    assert proposal.num_nodes == 1
    assert proposal.max_depth == 0
    assert jnp.array_equal(frontier.node_indices, jnp.asarray([[0], [0]], dtype=jnp.int32))
    assert jnp.array_equal(frontier.token_ids, jnp.asarray([[2], [3]], dtype=jnp.int32))
    assert jnp.array_equal(frontier.gumbel_positions, jnp.asarray([[7], [9]], dtype=jnp.int32))
    assert jnp.array_equal(frontier.gumbel_node_ids, jnp.zeros((batch_size, 1), dtype=jnp.int32))
    assert frontier.sampling_policy.token_counts is None
    assert proposal.base_token_counts is not None
    assert proposal.base_token_counts.shape == (batch_size, vocabulary_size)
    assert frontier.path_token_ids.shape == (batch_size, 1, 0)


def test_sample_top_k_branches_policy_state_for_child_frontier() -> None:
    vocabulary_size = 6
    base_policy = SamplingPolicy.init(frequency_penalty=1.0).with_empty_token_counts(vocabulary_size).broadcast(2)
    keys = jax.random.split(jax.random.key(1), 2)
    proposal, frontier = TrieProposal.create(
        root_ids=jnp.asarray([0, 0], dtype=jnp.int32),
        root_gumbel_positions=jnp.asarray([1, 1], dtype=jnp.int32),
        sampling_policy=base_policy,
        gumbel_keys=keys,
        vocabulary_size=vocabulary_size,
        budget=5,
    )
    logits = jnp.broadcast_to(jnp.arange(vocabulary_size, dtype=jnp.float32), (2, 1, vocabulary_size))

    sampled = frontier.sample_top_k(
        logits=logits,
        widths=jnp.asarray([[2], [1]], dtype=jnp.int32),
        max_width=2,
    )
    proposal, child_frontier = proposal.add_frontier(sampled)

    assert proposal.num_nodes == 3
    assert proposal.max_depth == 1
    assert jnp.array_equal(child_frontier.node_indices, jnp.asarray([[1, 2], [1, 0]], dtype=jnp.int32))
    assert jnp.array_equal(child_frontier.parent_indices, jnp.asarray([[0, 0], [0, -1]], dtype=jnp.int32))
    assert jnp.array_equal(child_frontier.gumbel_positions, jnp.asarray([[2, 2], [2, 0]], dtype=jnp.int32))
    assert jnp.array_equal(child_frontier.gumbel_node_ids, jnp.asarray([[1, 2], [1, 0]], dtype=jnp.int32))
    assert jnp.array_equal(child_frontier.mask, jnp.asarray([[True, True], [True, False]]))
    assert child_frontier.sampling_policy.token_counts is None
    assert child_frontier.base_token_counts is not None
    assert child_frontier.base_token_counts.shape == (2, vocabulary_size)
    assert jnp.array_equal(child_frontier.path_token_ids[:, :, 0], child_frontier.token_ids)
    assert jnp.array_equal(child_frontier.path_mask[:, :, 0], child_frontier.mask)


def test_target_sample_uses_per_node_gumbel_identity_in_one_batch() -> None:
    vocabulary_size = 7
    keys = jax.random.split(jax.random.key(2), 1)
    policy = SamplingPolicy.init().broadcast(1)
    proposal, frontier = TrieProposal.create(
        root_ids=jnp.asarray([0], dtype=jnp.int32),
        root_gumbel_positions=jnp.asarray([4], dtype=jnp.int32),
        sampling_policy=policy,
        gumbel_keys=keys,
        vocabulary_size=vocabulary_size,
        budget=4,
    )
    logits = jnp.zeros((1, 1, vocabulary_size), dtype=jnp.float32)
    sampled = frontier.sample_top_k(logits=logits, widths=jnp.asarray([[2]], dtype=jnp.int32), max_width=2)
    proposal, _child_frontier = proposal.add_frontier(sampled)

    target_logits = jnp.zeros((1, proposal.num_nodes, vocabulary_size), dtype=jnp.float32)
    target_sample = proposal.all_nodes_frontier().sample_one(target_logits)
    expected = [
        jax.random.categorical(
            fold_gumbel_key(
                keys[0],
                proposal.gumbel_positions[0, node_index],
                proposal.gumbel_node_ids[0, node_index],
            ),
            target_logits[0, node_index],
        ).astype(jnp.int32)
        for node_index in range(3)
    ]

    assert jnp.array_equal(target_sample.token_ids[0, :3], jnp.asarray(expected, dtype=jnp.int32))
    assert target_sample.token_ids.shape == (1, proposal.num_nodes)


def test_root_only_verify_keeps_single_slot_result() -> None:
    keys = jax.random.split(jax.random.key(3), 2)
    policy = SamplingPolicy.init().broadcast(2)
    proposal, _frontier = TrieProposal.create(
        root_ids=jnp.asarray([1, 2], dtype=jnp.int32),
        root_gumbel_positions=jnp.asarray([1, 1], dtype=jnp.int32),
        sampling_policy=policy,
        gumbel_keys=keys,
        vocabulary_size=5,
        budget=1,
    )
    processed_logits, sampled_token_ids, next_policy = proposal.sample(jnp.zeros((2, 1, 5), dtype=jnp.float32))

    accepted = proposal.verify(sampled_token_ids, next_policy)

    assert processed_logits.shape == (2, 1, 5)
    assert jnp.array_equal(accepted.accepted_token_ids, jnp.asarray([[1], [2]], dtype=jnp.int32))
    assert jnp.array_equal(accepted.compact_indices, jnp.asarray([[0], [0]], dtype=jnp.int32))
    assert jnp.array_equal(accepted.num_compact_indices, jnp.asarray([1, 1], dtype=jnp.int32))


def test_verify_uses_tree_depth_slots_not_budget_slots() -> None:
    vocabulary_size = 5
    keys = jax.random.split(jax.random.key(3), 1)
    policy = SamplingPolicy.init().broadcast(1)
    proposal, frontier = TrieProposal.create(
        root_ids=jnp.asarray([1], dtype=jnp.int32),
        root_gumbel_positions=jnp.asarray([1], dtype=jnp.int32),
        sampling_policy=policy,
        gumbel_keys=keys,
        vocabulary_size=vocabulary_size,
        budget=8,
    )
    logits = jnp.asarray([[[0.0, 1.0, 2.0, 3.0, 4.0]]], dtype=jnp.float32)
    sampled = frontier.sample_top_k(logits=logits, widths=jnp.asarray([[2]], dtype=jnp.int32), max_width=2)
    proposal, child_frontier = proposal.add_frontier(sampled)
    sampled_token_ids = jnp.full((1, proposal.budget), -1, dtype=jnp.int32)
    sampled_token_ids = sampled_token_ids.at[0, 0].set(child_frontier.token_ids[0, 0])
    sampled_token_ids = sampled_token_ids.at[0, child_frontier.node_indices[0, 0]].set(4)
    next_policy = proposal.all_nodes_frontier().sampling_policy

    accepted = proposal.verify(sampled_token_ids, next_policy)

    assert accepted.accepted_token_ids.shape == (1, 2)
    assert jnp.array_equal(
        accepted.accepted_token_ids,
        jnp.asarray([[1, child_frontier.token_ids[0, 0]]], dtype=jnp.int32),
    )
    assert jnp.array_equal(
        accepted.compact_indices,
        jnp.asarray([[0, child_frontier.node_indices[0, 0]]], dtype=jnp.int32),
    )
    assert jnp.array_equal(accepted.num_compact_indices, jnp.asarray([2], dtype=jnp.int32))
    assert jnp.array_equal(accepted.bonus_token_ids, jnp.asarray([4], dtype=jnp.int32))


def test_forward_inputs_are_sliced_to_static_node_count() -> None:
    vocabulary_size = 5
    keys = jax.random.split(jax.random.key(4), 1)
    proposal, frontier = TrieProposal.create(
        root_ids=jnp.asarray([1], dtype=jnp.int32),
        root_gumbel_positions=jnp.asarray([1], dtype=jnp.int32),
        sampling_policy=SamplingPolicy.init().broadcast(1),
        gumbel_keys=keys,
        vocabulary_size=vocabulary_size,
        budget=8,
    )
    sampled = frontier.sample_top_k(
        logits=jnp.zeros((1, 1, vocabulary_size), dtype=jnp.float32),
        widths=jnp.asarray([[2]], dtype=jnp.int32),
        max_width=2,
    )
    proposal, _frontier = proposal.add_frontier(sampled)

    inputs = proposal.forward_inputs(jnp.asarray([10], dtype=jnp.int32))

    assert proposal.num_nodes == 3
    assert inputs.token_ids.shape == (1, 3)
    assert inputs.token_positions.shape == (1, 3)
    assert inputs.attention_parent_indices is not None
    assert inputs.attention_parent_indices.shape == (1, 3)


def test_width_zero_creates_no_valid_children() -> None:
    vocabulary_size = 5
    keys = jax.random.split(jax.random.key(4), 1)
    proposal, frontier = TrieProposal.create(
        root_ids=jnp.asarray([0], dtype=jnp.int32),
        root_gumbel_positions=jnp.asarray([1], dtype=jnp.int32),
        sampling_policy=SamplingPolicy.init().broadcast(1),
        gumbel_keys=keys,
        vocabulary_size=vocabulary_size,
        budget=4,
    )

    sampled = frontier.sample_top_k(
        logits=jnp.zeros((1, 1, vocabulary_size), dtype=jnp.float32),
        widths=jnp.asarray([[0]], dtype=jnp.int32),
        max_width=3,
    )
    proposal, child_frontier = proposal.add_frontier(sampled)

    assert proposal.num_nodes == 4
    assert not bool(jnp.any(child_frontier.mask))
    assert not bool(jnp.any(proposal.node_mask[:, 1:]))


def test_add_frontier_masks_children_beyond_budget_without_raising() -> None:
    vocabulary_size = 6
    keys = jax.random.split(jax.random.key(5), 1)
    proposal, frontier = TrieProposal.create(
        root_ids=jnp.asarray([0], dtype=jnp.int32),
        root_gumbel_positions=jnp.asarray([1], dtype=jnp.int32),
        sampling_policy=SamplingPolicy.init().broadcast(1),
        gumbel_keys=keys,
        vocabulary_size=vocabulary_size,
        budget=3,
    )

    sampled = frontier.sample_top_k(
        logits=jnp.zeros((1, 1, vocabulary_size), dtype=jnp.float32),
        widths=jnp.asarray([[4]], dtype=jnp.int32),
        max_width=4,
    )
    proposal, child_frontier = proposal.add_frontier(sampled)

    assert proposal.num_nodes == proposal.budget
    assert jnp.array_equal(child_frontier.mask, jnp.asarray([[True, True, False, False]]))
    assert jnp.array_equal(proposal.node_mask, jnp.asarray([[True, True, True]]))
