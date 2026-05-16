from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int, Key  # noqa: TC002

from lalamo.module import ForwardPassMode
from lalamo.modules.utils import call_vmapped
from lalamo.sampling import SamplingPolicy

__all__ = ["AcceptedProposal", "ProposalInputs", "TrieProposal"]


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


class TrieProposal(eqx.Module):
    token_ids: Int[Array, "batch nodes"]
    parent_indices: Int[Array, "batch nodes"]
    depths: Int[Array, "batch nodes"]
    sample_positions: Int[Array, "batch nodes"]
    node_mask: Bool[Array, "batch nodes"]
    draft_policies: SamplingPolicy
    gumbel_keys: Key[Array, " batch"]
    cursor_indices: Int[Array, " batch"]
    vocabulary_size: int = eqx.field(static=True)
    num_nodes: int = eqx.field(static=True)

    @staticmethod
    def create(
        root_ids: Int[Array, " batch"],
        root_sample_positions: Int[Array, " batch"],
        sampling_policy: SamplingPolicy,
        gumbel_keys: Key[Array, " batch"],
        vocabulary_size: int,
        budget: int = 128,
    ) -> TrieProposal:
        (batch_size,) = root_ids.shape
        token_ids = jnp.zeros((batch_size, budget), dtype=jnp.int32)
        parent_indices = jnp.full((batch_size, budget), -1, dtype=jnp.int32)
        depths = jnp.zeros((batch_size, budget), dtype=jnp.int32)
        sample_positions = jnp.zeros((batch_size, budget), dtype=jnp.int32)
        node_mask = jnp.zeros((batch_size, budget), dtype=jnp.bool)
        draft_policies = jax.tree.map(
            lambda value: jnp.broadcast_to(value[:, None], (batch_size, budget, *value.shape[1:]))
            if eqx.is_array(value)
            else value,
            sampling_policy,
        )
        return TrieProposal(
            token_ids=token_ids.at[:, 0].set(root_ids),
            parent_indices=parent_indices,
            depths=depths,
            sample_positions=sample_positions.at[:, 0].set(root_sample_positions),
            node_mask=node_mask.at[:, 0].set(True),
            draft_policies=draft_policies,
            gumbel_keys=gumbel_keys,
            cursor_indices=jnp.zeros((batch_size,), dtype=jnp.int32),
            vocabulary_size=vocabulary_size,
            num_nodes=1,
        )

    @property
    def batch_size(self) -> int:
        return self.token_ids.shape[0]

    @property
    def budget(self) -> int:
        return self.token_ids.shape[1]

    def add_nodes(
        self,
        logits: Float[Array, "active vocabulary"],
        width: int = 1,
        batch_indices: Int[Array, " active"] | None = None,
        parent_indices: Int[Array, " active"] | None = None,
    ) -> tuple[TrieProposal, int]:
        assert logits.shape[-1] == self.vocabulary_size
        should_advance_cursor = parent_indices is None
        if batch_indices is None:
            batch_indices = jnp.arange(self.batch_size, dtype=jnp.int32)
        if parent_indices is None:
            parent_indices = self.cursor_indices[batch_indices]
        token_ids = self.sample_token_ids(batch_indices, parent_indices, logits, width)

        node_index = self.num_nodes
        parent_draft_policies = self.draft_policies_at(batch_indices, parent_indices)
        proposal = self
        for rank in range(token_ids.shape[1]):
            current_node_index = node_index + rank
            node_draft_policies = call_vmapped(
                lambda policy, token_id: policy.with_next_token_count(token_id),
                parent_draft_policies,
                token_ids[:, rank],
            )
            draft_policies = proposal.with_draft_policies(batch_indices, current_node_index, node_draft_policies)
            proposal = TrieProposal(
                token_ids=proposal.token_ids.at[batch_indices, current_node_index].set(token_ids[:, rank]),
                parent_indices=proposal.parent_indices.at[batch_indices, current_node_index].set(parent_indices),
                depths=proposal.depths.at[batch_indices, current_node_index].set(
                    proposal.depths[batch_indices, parent_indices] + 1,
                ),
                sample_positions=proposal.sample_positions.at[batch_indices, current_node_index].set(
                    proposal.sample_positions[batch_indices, parent_indices] + 1,
                ),
                node_mask=proposal.node_mask.at[batch_indices, current_node_index].set(True),
                draft_policies=draft_policies,
                gumbel_keys=proposal.gumbel_keys,
                cursor_indices=proposal.cursor_indices,
                vocabulary_size=proposal.vocabulary_size,
                num_nodes=current_node_index + 1,
            )
        if should_advance_cursor:
            proposal = TrieProposal(
                token_ids=proposal.token_ids,
                parent_indices=proposal.parent_indices,
                depths=proposal.depths,
                sample_positions=proposal.sample_positions,
                node_mask=proposal.node_mask,
                draft_policies=proposal.draft_policies,
                gumbel_keys=proposal.gumbel_keys,
                cursor_indices=proposal.cursor_indices.at[batch_indices].set(node_index),
                vocabulary_size=proposal.vocabulary_size,
                num_nodes=proposal.num_nodes,
            )
        return proposal, node_index

    def cursor_token_ids(self, batch_indices: Int[Array, " active"] | None = None) -> Int[Array, " active"]:
        if batch_indices is None:
            batch_indices = jnp.arange(self.batch_size, dtype=jnp.int32)
        return self.token_ids[batch_indices, self.cursor_indices[batch_indices]]

    def draft_policies_at(
        self,
        batch_indices: Int[Array, " active"],
        node_indices: Int[Array, " active"],
    ) -> SamplingPolicy:
        return jax.tree.map(
            lambda value: value[batch_indices, node_indices] if eqx.is_array(value) else value,
            self.draft_policies,
        )

    def with_draft_policies(
        self,
        batch_indices: Int[Array, " active"],
        node_index: int,
        draft_policies: SamplingPolicy,
    ) -> SamplingPolicy:
        return jax.tree.map(
            lambda values, value: values.at[batch_indices, node_index].set(value) if eqx.is_array(values) else values,
            self.draft_policies,
            draft_policies,
        )

    def sample_token_ids(
        self,
        batch_indices: Int[Array, " active"],
        parent_indices: Int[Array, " active"],
        logits: Float[Array, "active vocabulary"],
        width: int,
    ) -> Int[Array, "active width"]:
        width = min(width, logits.shape[-1], self.budget - self.num_nodes)
        if width < 1:
            raise ValueError("width must be at least 1 and fit in the proposal budget.")

        policies = self.draft_policies_at(batch_indices, parent_indices)
        positions = self.sample_positions[batch_indices, parent_indices]
        keys = jax.vmap(lambda key, position: jax.random.fold_in(key, position.astype(jnp.int32)))(
            self.gumbel_keys[batch_indices],
            positions,
        )
        return jax.vmap(
            lambda policy, key, row_logits: policy.sample_top_k(row_logits, key, width),
        )(policies, keys, logits)

    def sample(
        self,
        logits: Float[Array, "batch nodes vocabulary"],
    ) -> tuple[Float[Array, "batch nodes vocabulary"], Int[Array, "batch nodes"], SamplingPolicy]:
        assert logits.shape[-1] == self.vocabulary_size

        def sample_node(
            policy: SamplingPolicy,
            position: Int[Array, ""],
            base_key: Key[Array, ""],
            node_logits: Float[Array, " vocabulary"],
            node_mask: Bool[Array, ""],
        ) -> tuple[Float[Array, " vocabulary"], Int[Array, ""], SamplingPolicy]:
            key = jax.random.fold_in(base_key, position.astype(jnp.int32))
            processed_logits, token_id, next_policy = policy.sample(node_logits, key, node_mask)
            return jnp.where(node_mask, processed_logits, jnp.zeros_like(processed_logits)), token_id, next_policy

        def sample_row(
            policies: SamplingPolicy,
            row_positions: Int[Array, " nodes"],
            base_key: Key[Array, ""],
            row_logits: Float[Array, "nodes vocabulary"],
            row_mask: Bool[Array, " nodes"],
        ) -> tuple[Float[Array, "nodes vocabulary"], Int[Array, " nodes"], SamplingPolicy]:
            return call_vmapped(
                sample_node,
                policies,
                row_positions,
                jnp.broadcast_to(base_key, row_positions.shape),
                row_logits,
                row_mask,
            )

        processed_logits, token_ids, next_sampling_policies = call_vmapped(
            sample_row,
            self.draft_policies,
            self.sample_positions,
            self.gumbel_keys,
            logits,
            self.node_mask,
        )
        return processed_logits, jnp.where(self.node_mask, token_ids, -1), next_sampling_policies

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

        def sampling_policy_at(
            terminal_node_indices: Int[Array, " batch"],
        ) -> SamplingPolicy:
            return jax.tree.map(
                lambda value: value[batch_indices, terminal_node_indices] if eqx.is_array(value) else value,
                next_sampling_policies,
            )

        if self.num_nodes == 1:
            num_compact_indices = jnp.ones((self.batch_size,), dtype=jnp.int32)
            compact_indices = jnp.zeros((self.batch_size, 1), dtype=jnp.int32)
            terminal_node_indices = jnp.zeros((self.batch_size,), dtype=jnp.int32)
            bonus_token_ids = sampled_token_ids[:, 0]
            return AcceptedProposal(
                accepted_token_ids=self.token_ids[:, :1],
                node_indices=jnp.zeros((self.batch_size, 1), dtype=jnp.int32),
                compact_indices=compact_indices,
                num_compact_indices=num_compact_indices,
                terminal_node_indices=terminal_node_indices,
                bonus_token_ids=bonus_token_ids,
                next_sampling_policy=sampling_policy_at(terminal_node_indices),
            )

        candidate_node_indices = jnp.arange(self.budget, dtype=jnp.int32)[None, :]

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
                self.node_mask,
                jnp.logical_and(
                    candidate_node_indices > 0,
                    jnp.logical_and(
                        self.parent_indices == terminal_node_indices[:, None],
                        self.token_ids == sampled_at_terminal[:, None],
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
            length=self.budget - 1,
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
        raw_num_compact_indices = jnp.sum(path_mask, axis=1).astype(jnp.int32) + 1
        slots = jnp.arange(self.budget, dtype=jnp.int32)[None, :]
        accepted_token_ids = jnp.where(
            slots < raw_num_compact_indices[:, None],
            jnp.take_along_axis(self.token_ids, raw_compact_indices, axis=1),
            -1,
        )
        compact_indices = raw_compact_indices
        num_compact_indices = raw_num_compact_indices
        bonus_token_ids = sampled_token_ids[batch_indices, terminal_node_indices]
        node_indices = jnp.concatenate(
            [compact_indices[:, 1:], jnp.zeros((self.batch_size, 1), dtype=jnp.int32)],
            axis=1,
        )

        return AcceptedProposal(
            accepted_token_ids=accepted_token_ids,
            node_indices=node_indices,
            compact_indices=compact_indices,
            num_compact_indices=num_compact_indices,
            terminal_node_indices=terminal_node_indices,
            bonus_token_ids=bonus_token_ids,
            next_sampling_policy=sampling_policy_at(terminal_node_indices),
        )
