from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int, Key  # noqa: TC002

from lalamo.module import ForwardPassMode
from lalamo.sampling import SamplingPolicy

__all__ = [
    "AcceptedProposal",
    "Frontier",
    "ProposalInputs",
    "SampledFrontier",
    "TargetSample",
    "TrieProposal",
    "fold_gumbel_key",
]


class AcceptedProposal(eqx.Module):
    accepted_token_ids: Int[Array, "batch max_slots"]
    node_indices: Int[Array, "batch max_slots"]
    compact_indices: Int[Array, "batch max_slots"]
    num_compact_indices: Int[Array, " batch"]
    terminal_node_indices: Int[Array, " batch"]
    bonus_token_ids: Int[Array, " batch"]
    next_sampling_policy: SamplingPolicy

    def accepted_token_logits(
        self,
        processed_tree_logits: Float[Array, "batch nodes vocabulary"],
        root_sample_logits: Float[Array, "batch vocabulary"],
    ) -> Float[Array, "batch max_slots vocabulary"]:
        batch_size, _ = self.compact_indices.shape
        batch_indices = jnp.arange(batch_size, dtype=jnp.int32)[:, None]
        parent_logit_indices = jnp.concatenate(
            [jnp.zeros((batch_size, 1), dtype=jnp.int32), self.compact_indices[:, :-1]],
            axis=1,
        )
        token_logits = processed_tree_logits[batch_indices, parent_logit_indices]
        return token_logits.at[:, 0].set(root_sample_logits)

    def truncate(
        self,
        current_output_lengths: Int[Array, " batch"],
        max_output_length: int,
        done: Bool[Array, " batch"],
        eos_token_ids: Int[Array, " eos_tokens"],
    ) -> tuple[AcceptedProposal, Bool[Array, "batch max_slots"]]:
        slots = jnp.arange(self.accepted_token_ids.shape[1], dtype=jnp.int32)[None, :]
        valid = jnp.logical_and(
            slots < self.num_compact_indices[:, None],
            slots < (max_output_length - current_output_lengths)[:, None],
        )
        valid = jnp.logical_and(valid, jnp.logical_not(done)[:, None])
        if eos_token_ids.shape[0] > 0:
            eos_hits = jnp.logical_and(
                valid,
                jnp.any(self.accepted_token_ids[:, :, None] == eos_token_ids[None, None, :], axis=-1),
            )
        else:
            eos_hits = jnp.zeros_like(valid)
        prior_eos_count = jnp.cumsum(eos_hits.astype(jnp.int32), axis=1) - eos_hits.astype(jnp.int32)
        mask = jnp.logical_and(valid, prior_eos_count == 0)
        num_compact_indices = jnp.sum(mask, axis=1).astype(jnp.int32)
        return eqx.tree_at(lambda proposal: proposal.num_compact_indices, self, num_compact_indices), mask


class ProposalInputs(eqx.Module):
    token_ids: Int[Array, "batch nodes"]
    token_positions: Int[Array, "batch nodes"]
    lengths_without_padding: Int[Array, " batch"]
    forward_pass_mode: ForwardPassMode = eqx.field(static=True)
    attention_parent_indices: Int[Array, "batch nodes"] | None = None


class TargetSample(eqx.Module):
    processed_logits: Float[Array, "batch active vocabulary"]
    token_ids: Int[Array, "batch active"]
    next_sampling_policy: SamplingPolicy
    mask: Bool[Array, "batch active"]


class SampledFrontier(eqx.Module):
    processed_logits: Float[Array, "batch active vocabulary"]
    parent_indices: Int[Array, "batch active width"]
    token_ids: Int[Array, "batch active width"]
    depths: Int[Array, "batch active width"]
    gumbel_positions: Int[Array, "batch active width"]
    mask: Bool[Array, "batch active width"]
    sampling_policy: SamplingPolicy


class Frontier(eqx.Module):
    node_indices: Int[Array, "batch active"]
    parent_indices: Int[Array, "batch active"]
    token_ids: Int[Array, "batch active"]
    depths: Int[Array, "batch active"]
    gumbel_positions: Int[Array, "batch active"]
    gumbel_node_ids: Int[Array, "batch active"]
    mask: Bool[Array, "batch active"]
    sampling_policy: SamplingPolicy
    gumbel_keys: Key[Array, " batch"]

    def sample_one(self, logits: Float[Array, "batch active vocabulary"]) -> TargetSample:
        batch_size, active_size, vocabulary_size = logits.shape
        if self.node_indices.shape != (batch_size, active_size):
            raise ValueError("logits leading dimensions must match frontier shape.")

        flat_policy = self.sampling_policy.reshape((batch_size, active_size), (batch_size * active_size,))
        flat_keys = jnp.broadcast_to(self.gumbel_keys[:, None], (batch_size, active_size)).reshape(
            batch_size * active_size,
        )
        flat_positions = self.gumbel_positions.reshape(batch_size * active_size)
        flat_node_ids = self.gumbel_node_ids.reshape(batch_size * active_size)
        flat_logits = logits.reshape(batch_size * active_size, vocabulary_size)
        flat_mask = self.mask.reshape(batch_size * active_size)

        processed_logits, token_ids, next_policy = jax.vmap(sample_one_with_policy)(
            flat_policy,
            flat_logits,
            flat_keys,
            flat_positions,
            flat_node_ids,
            flat_mask,
        )
        return TargetSample(
            processed_logits=processed_logits.reshape(batch_size, active_size, vocabulary_size),
            token_ids=token_ids.reshape(batch_size, active_size),
            next_sampling_policy=next_policy.reshape((batch_size * active_size,), (batch_size, active_size)),
            mask=self.mask,
        )

    def sample_top_k(
        self,
        logits: Float[Array, "batch active vocabulary"],
        widths: Int[Array, "batch active"],
        max_width: int,
    ) -> SampledFrontier:
        batch_size, active_size, vocabulary_size = logits.shape
        if self.node_indices.shape != (batch_size, active_size):
            raise ValueError("logits leading dimensions must match frontier shape.")
        if widths.shape != (batch_size, active_size):
            raise ValueError("widths must match frontier shape.")
        if max_width < 1 or max_width > vocabulary_size:
            raise ValueError("max_width must be between 1 and vocabulary size.")

        flat_policy = self.sampling_policy.reshape((batch_size, active_size), (batch_size * active_size,))
        flat_keys = jnp.broadcast_to(self.gumbel_keys[:, None], (batch_size, active_size)).reshape(
            batch_size * active_size,
        )
        flat_positions = self.gumbel_positions.reshape(batch_size * active_size)
        flat_node_ids = self.gumbel_node_ids.reshape(batch_size * active_size)
        flat_logits = logits.reshape(batch_size * active_size, vocabulary_size)
        flat_widths = jnp.clip(widths, 0, max_width).reshape(batch_size * active_size)
        flat_mask = self.mask.reshape(batch_size * active_size)

        processed_logits, token_ids, child_policy, child_mask = jax.vmap(
            sample_top_k_with_policy,
            in_axes=(0, 0, 0, 0, 0, 0, 0, None),
        )(
            flat_policy,
            flat_logits,
            flat_keys,
            flat_positions,
            flat_node_ids,
            flat_mask,
            flat_widths,
            max_width,
        )
        return SampledFrontier(
            processed_logits=processed_logits.reshape(batch_size, active_size, vocabulary_size),
            parent_indices=jnp.broadcast_to(self.node_indices[:, :, None], (batch_size, active_size, max_width)),
            token_ids=token_ids.reshape(batch_size, active_size, max_width),
            depths=jnp.broadcast_to(self.depths[:, :, None] + 1, (batch_size, active_size, max_width)),
            gumbel_positions=jnp.broadcast_to(
                self.gumbel_positions[:, :, None] + 1,
                (batch_size, active_size, max_width),
            ),
            mask=child_mask.reshape(batch_size, active_size, max_width),
            sampling_policy=child_policy.reshape(
                (batch_size * active_size, max_width),
                (batch_size, active_size, max_width),
            ),
        )

    def take(self, active_indices: Int[Array, " next_active"]) -> Frontier:
        return Frontier(
            node_indices=self.node_indices[:, active_indices],
            parent_indices=self.parent_indices[:, active_indices],
            token_ids=self.token_ids[:, active_indices],
            depths=self.depths[:, active_indices],
            gumbel_positions=self.gumbel_positions[:, active_indices],
            gumbel_node_ids=self.gumbel_node_ids[:, active_indices],
            mask=self.mask[:, active_indices],
            sampling_policy=jax.tree.map(
                lambda value: value[:, active_indices] if eqx.is_array(value) else value,
                self.sampling_policy,
            ),
            gumbel_keys=self.gumbel_keys,
        )


class TrieProposal(eqx.Module):
    token_ids: Int[Array, "batch nodes"]
    parent_indices: Int[Array, "batch nodes"]
    depths: Int[Array, "batch nodes"]
    gumbel_positions: Int[Array, "batch nodes"]
    gumbel_node_ids: Int[Array, "batch nodes"]
    node_mask: Bool[Array, "batch nodes"]
    sampling_policies: SamplingPolicy
    gumbel_keys: Key[Array, " batch"]
    vocabulary_size: int = eqx.field(static=True)
    num_nodes: int = eqx.field(static=True)
    max_depth: int = eqx.field(static=True)

    @staticmethod
    def create(
        root_ids: Int[Array, " batch"],
        root_gumbel_positions: Int[Array, " batch"],
        sampling_policy: SamplingPolicy,
        gumbel_keys: Key[Array, " batch"],
        vocabulary_size: int,
        budget: int = 128,
    ) -> tuple[TrieProposal, Frontier]:
        (batch_size,) = root_ids.shape
        token_ids = jnp.zeros((batch_size, budget), dtype=jnp.int32)
        parent_indices = jnp.full((batch_size, budget), -1, dtype=jnp.int32)
        depths = jnp.zeros((batch_size, budget), dtype=jnp.int32)
        gumbel_positions = jnp.zeros((batch_size, budget), dtype=jnp.int32)
        gumbel_node_ids = jnp.zeros((batch_size, budget), dtype=jnp.int32)
        node_mask = jnp.zeros((batch_size, budget), dtype=jnp.bool)
        sampling_policies = jax.tree.map(
            lambda value: (
                jnp.broadcast_to(value[:, None], (batch_size, budget, *value.shape[1:]))
                if eqx.is_array(value)
                else value
            ),
            sampling_policy,
        )
        proposal = TrieProposal(
            token_ids=token_ids.at[:, 0].set(root_ids),
            parent_indices=parent_indices,
            depths=depths,
            gumbel_positions=gumbel_positions.at[:, 0].set(root_gumbel_positions),
            gumbel_node_ids=gumbel_node_ids,
            node_mask=node_mask.at[:, 0].set(True),
            sampling_policies=sampling_policies,
            gumbel_keys=gumbel_keys,
            vocabulary_size=vocabulary_size,
            num_nodes=1,
            max_depth=0,
        )
        frontier = Frontier(
            node_indices=jnp.zeros((batch_size, 1), dtype=jnp.int32),
            parent_indices=jnp.full((batch_size, 1), -1, dtype=jnp.int32),
            token_ids=root_ids[:, None],
            depths=jnp.zeros((batch_size, 1), dtype=jnp.int32),
            gumbel_positions=root_gumbel_positions[:, None],
            gumbel_node_ids=jnp.zeros((batch_size, 1), dtype=jnp.int32),
            mask=jnp.ones((batch_size, 1), dtype=jnp.bool),
            sampling_policy=jax.tree.map(
                lambda value: value[:, None] if eqx.is_array(value) else value,
                sampling_policy,
            ),
            gumbel_keys=gumbel_keys,
        )
        return proposal, frontier

    @property
    def batch_size(self) -> int:
        return self.token_ids.shape[0]

    @property
    def budget(self) -> int:
        return self.token_ids.shape[1]

    def add_frontier(self, sampled: SampledFrontier) -> tuple[TrieProposal, Frontier]:
        batch_size, active_size, max_width = sampled.token_ids.shape
        if batch_size != self.batch_size:
            raise ValueError("sampled frontier batch size must match proposal batch size.")
        child_slots = active_size * max_width
        remaining_slots = max(self.budget - self.num_nodes, 0)
        slot_mask = jnp.arange(child_slots, dtype=jnp.int32) < remaining_slots

        child_node_indices = jnp.arange(self.num_nodes, self.num_nodes + child_slots, dtype=jnp.int32)
        batch_indices = jnp.arange(batch_size, dtype=jnp.int32)[:, None]
        node_indices = child_node_indices[None, :]
        token_ids = sampled.token_ids.reshape(batch_size, child_slots)
        parent_indices = sampled.parent_indices.reshape(batch_size, child_slots)
        depths = sampled.depths.reshape(batch_size, child_slots)
        gumbel_positions = sampled.gumbel_positions.reshape(batch_size, child_slots)
        mask = sampled.mask.reshape(batch_size, child_slots) & slot_mask[None, :]
        gumbel_node_ids = jnp.broadcast_to(node_indices, (batch_size, child_slots))
        sampling_policy = sampled.sampling_policy.reshape(
            (batch_size, active_size, max_width),
            (batch_size, child_slots),
        )
        sampling_policies = jax.tree.map(
            lambda values, updates: (
                values.at[batch_indices, node_indices].set(updates, mode="drop") if eqx.is_array(values) else values
            ),
            self.sampling_policies,
            sampling_policy,
        )
        proposal = TrieProposal(
            token_ids=self.token_ids.at[batch_indices, node_indices].set(
                jnp.where(mask, token_ids, 0),
                mode="drop",
            ),
            parent_indices=self.parent_indices.at[batch_indices, node_indices].set(
                jnp.where(mask, parent_indices, -1),
                mode="drop",
            ),
            depths=self.depths.at[batch_indices, node_indices].set(jnp.where(mask, depths, 0), mode="drop"),
            gumbel_positions=self.gumbel_positions.at[batch_indices, node_indices].set(
                jnp.where(mask, gumbel_positions, 0),
                mode="drop",
            ),
            gumbel_node_ids=self.gumbel_node_ids.at[batch_indices, node_indices].set(
                jnp.where(mask, gumbel_node_ids, 0),
                mode="drop",
            ),
            node_mask=self.node_mask.at[batch_indices, node_indices].set(mask, mode="drop"),
            sampling_policies=sampling_policies,
            gumbel_keys=self.gumbel_keys,
            vocabulary_size=self.vocabulary_size,
            num_nodes=min(self.num_nodes + child_slots, self.budget),
            max_depth=self.max_depth + 1,
        )
        child_frontier = Frontier(
            node_indices=jnp.where(mask, jnp.broadcast_to(node_indices, (batch_size, child_slots)), 0),
            parent_indices=jnp.where(mask, parent_indices, -1),
            token_ids=jnp.where(mask, token_ids, 0),
            depths=jnp.where(mask, depths, 0),
            gumbel_positions=jnp.where(mask, gumbel_positions, 0),
            gumbel_node_ids=jnp.where(mask, gumbel_node_ids, 0),
            mask=mask,
            sampling_policy=sampling_policy,
            gumbel_keys=self.gumbel_keys,
        )
        return proposal, child_frontier

    def all_nodes_frontier(self) -> Frontier:
        node_indices = jnp.broadcast_to(
            jnp.arange(self.budget, dtype=jnp.int32)[None, :],
            (self.batch_size, self.budget),
        )
        return Frontier(
            node_indices=node_indices,
            parent_indices=self.parent_indices,
            token_ids=self.token_ids,
            depths=self.depths,
            gumbel_positions=self.gumbel_positions,
            gumbel_node_ids=self.gumbel_node_ids,
            mask=self.node_mask,
            sampling_policy=self.sampling_policies,
            gumbel_keys=self.gumbel_keys,
        )

    def sample(
        self,
        logits: Float[Array, "batch nodes vocabulary"],
    ) -> tuple[Float[Array, "batch nodes vocabulary"], Int[Array, "batch nodes"], SamplingPolicy]:
        if logits.shape[-1] != self.vocabulary_size:
            raise ValueError("logits vocabulary dimension must match proposal vocabulary size.")
        target_sample = self.all_nodes_frontier().sample_one(logits)
        return target_sample.processed_logits, target_sample.token_ids, target_sample.next_sampling_policy

    def forward_inputs(
        self,
        next_token_positions: Int[Array, " batch"],
    ) -> ProposalInputs:
        token_positions = next_token_positions[:, None] + self.depths
        forward_pass_mode = ForwardPassMode.MULTI_TOKEN
        if self.num_nodes == 1:
            forward_pass_mode = ForwardPassMode.SINGLE_TOKEN
        attention_parent_indices = None
        if self.num_nodes > 1:
            attention_parent_indices = jnp.where(self.node_mask, self.parent_indices, -1)
        return ProposalInputs(
            token_ids=self.token_ids,
            token_positions=token_positions,
            lengths_without_padding=jnp.full((self.batch_size,), self.num_nodes, dtype=jnp.int32),
            forward_pass_mode=forward_pass_mode,
            attention_parent_indices=attention_parent_indices,
        )

    def verify(
        self,
        sampled_token_ids: Int[Array, "batch nodes"],
        next_sampling_policies: SamplingPolicy,
    ) -> AcceptedProposal:
        batch_indices = jnp.arange(self.batch_size, dtype=jnp.int32)
        candidate_node_indices = jnp.arange(self.num_nodes, dtype=jnp.int32)[None, :]
        node_mask = self.node_mask[:, : self.num_nodes]
        parent_indices = self.parent_indices[:, : self.num_nodes]
        token_ids = self.token_ids[:, : self.num_nodes]

        def sampling_policy_at(
            terminal_node_indices: Int[Array, " batch"],
        ) -> SamplingPolicy:
            return jax.tree.map(
                lambda value: value[batch_indices, terminal_node_indices] if eqx.is_array(value) else value,
                next_sampling_policies,
            )

        def scan_step(
            carry: tuple[Int[Array, " batch"], Bool[Array, " batch"]],
            _: None,
        ) -> tuple[
            tuple[Int[Array, " batch"], Bool[Array, " batch"]],
            tuple[Int[Array, " batch"], Bool[Array, " batch"]],
        ]:
            terminal_node_indices, alive = carry
            sampled_at_terminal = sampled_token_ids[batch_indices, terminal_node_indices]
            child_mask = jnp.logical_and(
                node_mask,
                jnp.logical_and(
                    candidate_node_indices > 0,
                    jnp.logical_and(
                        parent_indices == terminal_node_indices[:, None],
                        token_ids == sampled_at_terminal[:, None],
                    ),
                ),
            )
            accepted = jnp.logical_and(alive, jnp.any(child_mask, axis=1))
            child_indices = jnp.argmax(child_mask, axis=1).astype(jnp.int32)
            next_terminal_node_indices = jnp.where(accepted, child_indices, terminal_node_indices)
            return (next_terminal_node_indices, accepted), (child_indices, accepted)

        (terminal_node_indices, _), (path_node_indices, path_mask) = jax.lax.scan(
            scan_step,
            (jnp.zeros((self.batch_size,), dtype=jnp.int32), jnp.ones((self.batch_size,), dtype=jnp.bool)),
            xs=None,
            length=self.max_depth,
        )

        path_node_indices = path_node_indices.T
        path_mask = path_mask.T
        raw_node_indices = jnp.concatenate(
            [
                jnp.where(path_mask, path_node_indices, 0),
                jnp.zeros((self.batch_size, 1), dtype=jnp.int32),
            ],
            axis=1,
        )
        raw_compact_indices = jnp.concatenate(
            [jnp.zeros((self.batch_size, 1), dtype=jnp.int32), raw_node_indices[:, :-1]],
            axis=1,
        )
        num_compact_indices = jnp.sum(path_mask, axis=1).astype(jnp.int32) + 1
        slots = jnp.arange(self.max_depth + 1, dtype=jnp.int32)[None, :]
        accepted_token_ids = jnp.where(
            slots < num_compact_indices[:, None],
            jnp.take_along_axis(self.token_ids, raw_compact_indices, axis=1),
            -1,
        )
        node_indices = jnp.concatenate(
            [raw_compact_indices[:, 1:], jnp.zeros((self.batch_size, 1), dtype=jnp.int32)],
            axis=1,
        )
        bonus_token_ids = sampled_token_ids[batch_indices, terminal_node_indices]
        return AcceptedProposal(
            accepted_token_ids=accepted_token_ids,
            node_indices=node_indices,
            compact_indices=raw_compact_indices,
            num_compact_indices=num_compact_indices,
            terminal_node_indices=terminal_node_indices,
            bonus_token_ids=bonus_token_ids,
            next_sampling_policy=sampling_policy_at(terminal_node_indices),
        )


def fold_gumbel_key(
    key: Key[Array, ""],
    position: Int[Array, ""],
    node_id: Int[Array, ""],
) -> Key[Array, ""]:
    key = jax.random.fold_in(key, position.astype(jnp.int32))
    return jax.random.fold_in(key, node_id.astype(jnp.int32))


def sample_one_with_policy(
    policy: SamplingPolicy,
    logits: Float[Array, " vocabulary"],
    key: Key[Array, ""],
    position: Int[Array, ""],
    node_id: Int[Array, ""],
    mask: Bool[Array, ""],
) -> tuple[Float[Array, " vocabulary"], Int[Array, ""], SamplingPolicy]:
    sample_key = fold_gumbel_key(key, position, node_id)
    safe_logits = jnp.where(mask, logits, jnp.zeros_like(logits))
    processed_logits, token_id, next_policy = policy.sample(safe_logits, sample_key, mask)
    return (
        jnp.where(mask, processed_logits, jnp.zeros_like(processed_logits)),
        jnp.where(mask, token_id, -1),
        next_policy,
    )


def sample_top_k_with_policy(
    policy: SamplingPolicy,
    logits: Float[Array, " vocabulary"],
    key: Key[Array, ""],
    position: Int[Array, ""],
    node_id: Int[Array, ""],
    mask: Bool[Array, ""],
    width: Int[Array, ""],
    max_width: int,
) -> tuple[Float[Array, " vocabulary"], Int[Array, " width"], SamplingPolicy, Bool[Array, " width"]]:
    sample_key = fold_gumbel_key(key, position, node_id)
    sample_mask = mask & (width > 0)
    safe_logits = jnp.where(sample_mask, logits, jnp.zeros_like(logits))
    processed_logits = policy.process_logits(safe_logits.astype(jnp.float32))
    token_ids = jax.random.categorical(sample_key, processed_logits, shape=(max_width,), replace=False).astype(
        jnp.int32,
    )
    child_mask = mask & (jnp.arange(max_width, dtype=jnp.int32) < width)

    def count_token(token_id: Int[Array, ""], should_count: Bool[Array, ""]) -> SamplingPolicy:
        return policy.with_next_token_count(token_id, should_count)

    child_policy = jax.vmap(count_token)(token_ids, child_mask)
    return (
        jnp.where(sample_mask, processed_logits, jnp.zeros_like(processed_logits)),
        token_ids,
        child_policy,
        child_mask,
    )
