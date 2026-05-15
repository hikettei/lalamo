from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int, Key  # noqa: TC002

from lalamo.module import ForwardPassMode
from lalamo.sampling import SamplingPolicy

__all__ = ["AcceptedProposal", "ProposalInputs", "TrieProposal"]


class AcceptedProposal(eqx.Module):
    accepted_token_ids: Int[Array, "batch max_slots"]
    node_indices: Int[Array, "batch max_slots"]
    compact_indices: Int[Array, "batch max_slots"]
    num_compact_indices: Int[Array, " batch"]
    bonus_token_ids: Int[Array, " batch"]
    terminal_sample_logits: Float[Array, "batch vocabulary"]
    sampling_top_k_ids: Int[Array, "batch max_slots k"] | None = None
    sampling_top_k_logits: Float[Array, "batch max_slots k"] | None = None

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
    num_nodes: int = eqx.field(static=True)

    @staticmethod
    def create(
        root_ids: Int[Array, " batch"],
        root_sample_positions: Int[Array, " batch"],
        budget: int = 128,
    ) -> TrieProposal:
        (batch_size,) = root_ids.shape
        token_ids = jnp.zeros((batch_size, budget), dtype=jnp.int32)
        parent_indices = jnp.full((batch_size, budget), -1, dtype=jnp.int32)
        depths = jnp.zeros((batch_size, budget), dtype=jnp.int32)
        sample_positions = jnp.zeros((batch_size, budget), dtype=jnp.int32)
        node_mask = jnp.zeros((batch_size, budget), dtype=jnp.bool)
        return TrieProposal(
            token_ids=token_ids.at[:, 0].set(root_ids),
            parent_indices=parent_indices,
            depths=depths,
            sample_positions=sample_positions.at[:, 0].set(root_sample_positions),
            node_mask=node_mask.at[:, 0].set(True),
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
        batch_indices: Int[Array, " active"],
        parent_indices: Int[Array, " active"],
        token_ids: Int[Array, " active"],
    ) -> tuple[TrieProposal, int]:
        node_index = self.num_nodes
        return (
            TrieProposal(
                token_ids=self.token_ids.at[batch_indices, node_index].set(token_ids),
                parent_indices=self.parent_indices.at[batch_indices, node_index].set(parent_indices),
                depths=self.depths.at[batch_indices, node_index].set(
                    self.depths[batch_indices, parent_indices] + 1,
                ),
                sample_positions=self.sample_positions.at[batch_indices, node_index].set(
                    self.sample_positions[batch_indices, parent_indices] + 1,
                ),
                node_mask=self.node_mask.at[batch_indices, node_index].set(True),
                num_nodes=node_index + 1,
            ),
            node_index,
        )

    def sample_and_verify(
        self,
        logits: Float[Array, "batch nodes vocabulary"],
        sampling_policy: SamplingPolicy,
        output_lengths: Int[Array, " batch"],
        per_position_keys: Key[Array, "batch positions"],
        root_sample_logits: Float[Array, "batch vocabulary"],
        num_top_logits_to_return: int | None,
    ) -> AcceptedProposal:
        batch_size, max_slots = self.token_ids.shape
        batch_indices = jnp.arange(batch_size, dtype=jnp.int32)
        candidate_node_indices = jnp.arange(max_slots, dtype=jnp.int32)[None, :]
        initial_policy = jax.vmap(
            lambda policy, token_id, should_count: policy.with_next_token_count(token_id, should_count),
        )(
            sampling_policy,
            self.token_ids[:, 0],
            self.node_mask[:, 0],
        )
        initial_carry = (
            initial_policy,
            jnp.zeros((batch_size,), dtype=jnp.int32),
            jnp.ones((batch_size,), dtype=jnp.bool),
            jnp.zeros((batch_size,), dtype=jnp.int32),
            jnp.zeros_like(logits[:, 0, :], dtype=jnp.float32),
        )

        def advance(
            carry: tuple[
                SamplingPolicy,
                Int[Array, " batch"],
                Bool[Array, " batch"],
                Int[Array, " batch"],
                Float[Array, "batch vocabulary"],
            ],
        ) -> tuple[
            tuple[
                SamplingPolicy,
                Int[Array, " batch"],
                Bool[Array, " batch"],
                Int[Array, " batch"],
                Float[Array, "batch vocabulary"],
            ],
            Int[Array, " batch"],
            Bool[Array, " batch"],
            Float[Array, "batch vocabulary"],
        ]:
            current_policy, terminal_node_indices, alive, bonus_token_ids, terminal_sample_logits = carry
            terminal_depths = self.depths[batch_indices, terminal_node_indices]
            sample_positions = output_lengths + terminal_depths + 1
            safe_positions = jnp.clip(sample_positions, 0, per_position_keys.shape[1] - 1)
            sample_keys = per_position_keys[batch_indices, safe_positions]
            terminal_logits = logits[batch_indices, terminal_node_indices]
            sampled_logits = jax.vmap(
                lambda policy, row_logits: policy.process_logits(row_logits.astype(jnp.float32))
            )(
                current_policy,
                terminal_logits,
            )
            sampled_token_ids = jax.vmap(jax.random.categorical)(sample_keys, sampled_logits).astype(jnp.int32)
            child_mask = (
                self.node_mask
                & (candidate_node_indices > 0)
                & (self.parent_indices == terminal_node_indices[:, None])
                & (self.token_ids == sampled_token_ids[:, None])
            )
            accepted = alive & jnp.any(child_mask, axis=1)
            child_indices = jnp.argmax(child_mask, axis=1).astype(jnp.int32)
            next_terminal_node_indices = jnp.where(accepted, child_indices, terminal_node_indices)
            next_policy = jax.vmap(
                lambda policy, token_id, should_count: policy.with_next_token_count(token_id, should_count),
            )(
                current_policy,
                self.token_ids[batch_indices, child_indices],
                accepted,
            )
            next_bonus_token_ids = jnp.where(alive, sampled_token_ids, bonus_token_ids)
            next_terminal_sample_logits = jnp.where(alive[:, None], sampled_logits, terminal_sample_logits)
            return (
                (
                    next_policy,
                    next_terminal_node_indices,
                    accepted,
                    next_bonus_token_ids,
                    next_terminal_sample_logits,
                ),
                child_indices,
                accepted,
                sampled_logits,
            )

        def step(
            carry: tuple[
                SamplingPolicy,
                Int[Array, " batch"],
                Bool[Array, " batch"],
                Int[Array, " batch"],
                Float[Array, "batch vocabulary"],
            ],
            _: None,
        ) -> tuple[
            tuple[
                SamplingPolicy,
                Int[Array, " batch"],
                Bool[Array, " batch"],
                Int[Array, " batch"],
                Float[Array, "batch vocabulary"],
            ],
            tuple[Int[Array, " batch"], Bool[Array, " batch"]],
        ]:
            def active_step(
                carry: tuple[
                    SamplingPolicy,
                    Int[Array, " batch"],
                    Bool[Array, " batch"],
                    Int[Array, " batch"],
                    Float[Array, "batch vocabulary"],
                ],
            ) -> tuple[
                tuple[
                    SamplingPolicy,
                    Int[Array, " batch"],
                    Bool[Array, " batch"],
                    Int[Array, " batch"],
                    Float[Array, "batch vocabulary"],
                ],
                tuple[Int[Array, " batch"], Bool[Array, " batch"]],
            ]:
                next_carry, child_indices, accepted, _sampled_logits = advance(carry)
                return next_carry, (child_indices, accepted)

            def inactive_step(
                carry: tuple[
                    SamplingPolicy,
                    Int[Array, " batch"],
                    Bool[Array, " batch"],
                    Int[Array, " batch"],
                    Float[Array, "batch vocabulary"],
                ],
            ) -> tuple[
                tuple[
                    SamplingPolicy,
                    Int[Array, " batch"],
                    Bool[Array, " batch"],
                    Int[Array, " batch"],
                    Float[Array, "batch vocabulary"],
                ],
                tuple[Int[Array, " batch"], Bool[Array, " batch"]],
            ]:
                return carry, (
                    jnp.zeros((batch_size,), dtype=jnp.int32),
                    jnp.zeros((batch_size,), dtype=jnp.bool),
                )

            return jax.lax.cond(jnp.any(carry[2]), active_step, inactive_step, carry)

        if num_top_logits_to_return is None:
            (
                (
                    _final_policy,
                    _terminal_node_indices,
                    _final_alive,
                    bonus_token_ids,
                    terminal_sample_logits,
                ),
                (path_node_indices, path_mask),
            ) = jax.lax.scan(
                step,
                initial_carry,
                xs=None,
                length=max_slots,
            )
            sampling_top_k_ids = None
            sampling_top_k_logits = None
        else:
            root_top_k_logits, root_top_k_ids = jax.lax.top_k(root_sample_logits, num_top_logits_to_return)

            def top_k_step(
                carry: tuple[
                    SamplingPolicy,
                    Int[Array, " batch"],
                    Bool[Array, " batch"],
                    Int[Array, " batch"],
                    Float[Array, "batch vocabulary"],
                ],
                _: None,
            ) -> tuple[
                tuple[
                    SamplingPolicy,
                    Int[Array, " batch"],
                    Bool[Array, " batch"],
                    Int[Array, " batch"],
                    Float[Array, "batch vocabulary"],
                ],
                tuple[
                    Int[Array, " batch"],
                    Bool[Array, " batch"],
                    Int[Array, "batch k"],
                    Float[Array, "batch k"],
                ],
            ]:
                def active_step(
                    carry: tuple[
                        SamplingPolicy,
                        Int[Array, " batch"],
                        Bool[Array, " batch"],
                        Int[Array, " batch"],
                        Float[Array, "batch vocabulary"],
                    ],
                ) -> tuple[
                    tuple[
                        SamplingPolicy,
                        Int[Array, " batch"],
                        Bool[Array, " batch"],
                        Int[Array, " batch"],
                        Float[Array, "batch vocabulary"],
                    ],
                    tuple[
                        Int[Array, " batch"],
                        Bool[Array, " batch"],
                        Int[Array, "batch k"],
                        Float[Array, "batch k"],
                    ],
                ]:
                    next_carry, child_indices, accepted, sampled_logits = advance(carry)
                    top_k_logits, top_k_ids = jax.lax.top_k(sampled_logits, num_top_logits_to_return)
                    top_k_ids = jnp.where(accepted[:, None], top_k_ids, jnp.zeros_like(top_k_ids))
                    top_k_logits = jnp.where(accepted[:, None], top_k_logits, jnp.zeros_like(top_k_logits))
                    return next_carry, (child_indices, accepted, top_k_ids, top_k_logits)

                def inactive_step(
                    carry: tuple[
                        SamplingPolicy,
                        Int[Array, " batch"],
                        Bool[Array, " batch"],
                        Int[Array, " batch"],
                        Float[Array, "batch vocabulary"],
                    ],
                ) -> tuple[
                    tuple[
                        SamplingPolicy,
                        Int[Array, " batch"],
                        Bool[Array, " batch"],
                        Int[Array, " batch"],
                        Float[Array, "batch vocabulary"],
                    ],
                    tuple[
                        Int[Array, " batch"],
                        Bool[Array, " batch"],
                        Int[Array, "batch k"],
                        Float[Array, "batch k"],
                    ],
                ]:
                    return carry, (
                        jnp.zeros((batch_size,), dtype=jnp.int32),
                        jnp.zeros((batch_size,), dtype=jnp.bool),
                        jnp.zeros((batch_size, num_top_logits_to_return), dtype=jnp.int32),
                        jnp.zeros((batch_size, num_top_logits_to_return), dtype=jnp.float32),
                    )

                return jax.lax.cond(jnp.any(carry[2]), active_step, inactive_step, carry)

            (
                (
                    _final_policy,
                    _terminal_node_indices,
                    _final_alive,
                    bonus_token_ids,
                    terminal_sample_logits,
                ),
                (path_node_indices, path_mask, path_top_k_ids, path_top_k_logits),
            ) = jax.lax.scan(
                top_k_step,
                initial_carry,
                xs=None,
                length=max_slots,
            )
            path_top_k_ids = jnp.swapaxes(path_top_k_ids, 0, 1)
            path_top_k_logits = jnp.swapaxes(path_top_k_logits, 0, 1)
            sampling_top_k_ids = jnp.concatenate(
                [root_top_k_ids[:, None, :], path_top_k_ids[:, :-1]],
                axis=1,
            )
            sampling_top_k_logits = jnp.concatenate(
                [root_top_k_logits[:, None, :], path_top_k_logits[:, :-1]],
                axis=1,
            )

        path_node_indices = jnp.swapaxes(path_node_indices, 0, 1)
        path_mask = jnp.swapaxes(path_mask, 0, 1)
        node_indices = jnp.where(path_mask, path_node_indices, 0)
        compact_indices = jnp.concatenate(
            [jnp.zeros((batch_size, 1), dtype=jnp.int32), node_indices[:, :-1]],
            axis=1,
        )
        num_compact_indices = jnp.sum(path_mask, axis=1).astype(jnp.int32) + 1
        slots = jnp.arange(max_slots, dtype=jnp.int32)[None, :]
        accepted_token_ids = jnp.where(
            slots < num_compact_indices[:, None],
            jnp.take_along_axis(self.token_ids, compact_indices, axis=1),
            -1,
        )
        return AcceptedProposal(
            accepted_token_ids=accepted_token_ids,
            node_indices=node_indices,
            compact_indices=compact_indices,
            num_compact_indices=num_compact_indices,
            bonus_token_ids=bonus_token_ids,
            terminal_sample_logits=terminal_sample_logits,
            sampling_top_k_ids=sampling_top_k_ids,
            sampling_top_k_logits=sampling_top_k_logits,
        )

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
