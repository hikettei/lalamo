import jax
import jax.numpy as jnp

from lalamo.modules.token_mixer import State
from lalamo.sampling import SamplingPolicy
from lalamo.speculator.proposal import TrieProposal
from lalamo.speculator.proposers.ngram import NGramModel, NGramSpeculator
from lalamo.speculator.state import LMState, MemoryBuffers, StateRequest


def test_ngram_token_id_zero_not_corrupted() -> None:
    model = NGramModel.init(256, 4, max_order=2)

    model.train([1, 0], [{1: 1.0}, {0: 1.0}])
    model.compress()

    probs = model.probs([1])
    assert 0 in probs, f"Token 0 missing from probs: {probs}"
    assert probs[0] > 0.0, f"Token 0 has zero probability: {probs}"

    total = sum(probs.values())
    assert abs(total - 1.0) < 0.01, f"Probs sum to {total}, expected ~1.0"


def test_ngram_serialize_roundtrip() -> None:
    model = NGramModel.init(512, 8, max_order=3, discount=0.01)

    token_ids = list(range(200))
    token_logits = [{k: 1.0} for k in token_ids]
    model.train(token_ids, token_logits)
    model.compress()

    blob = model.serialize()
    restored = NGramModel.deserialize(blob)

    assert blob == restored.serialize()


def draft_with_speculator(speculator: NGramSpeculator, state: LMState) -> TrieProposal:
    return speculator.draft(state)


def test_ngram_draft_uses_callback_tree_builder() -> None:
    model = NGramModel.init(256, 4, max_order=2)
    model.train([1, 2, 3], [{2: 1.0}, {3: 1.0}, {4: 1.0}])
    model.compress()
    speculator = NGramSpeculator.create(model, width=1, depth=2)
    state = LMState(
        kv_cache=State(()),
        next_token_position=jnp.asarray([1], dtype=jnp.int32),
        root_bonus_id=jnp.asarray([1], dtype=jnp.int32),
        root_sample_logits=jnp.zeros((1, 8), dtype=jnp.float32),
        sampling_policy=SamplingPolicy.init().broadcast(1),
        gumbel_keys=jax.random.split(jax.random.key(0), 1),
        output_lengths=jnp.asarray([0], dtype=jnp.int32),
        memory=MemoryBuffers.empty(StateRequest(token_id_capacity=1), batch_size=1, hidden_dim=4),
    )

    proposal = jax.jit(draft_with_speculator, static_argnums=0)(speculator, state)

    assert proposal.num_nodes == 3
    assert jnp.array_equal(proposal.token_ids[:, :3], jnp.asarray([[1, 3, 2]], dtype=jnp.int32))
    assert jnp.array_equal(proposal.parent_indices[:, :3], jnp.asarray([[-1, 0, 1]], dtype=jnp.int32))
