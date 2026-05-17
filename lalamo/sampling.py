from collections.abc import Iterable
from dataclasses import replace

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.typing import DTypeLike
from jaxtyping import Array, Bool, Float, Int, Key

from lalamo.module import Keychain

__all__ = ["SamplingPolicy"]


_SENTINEL = -1
_MAX_BANNED_TOKENS = 16
type SamplingLeaf = Float[Array, "..."] | Int[Array, "..."]


class SamplingPolicy(eqx.Module):
    temperature: Float[Array, "*batch"] | None = None
    top_k: Int[Array, "*batch"] | None = None
    top_k_limit: int | None = eqx.field(static=True, default=None)
    top_p: Float[Array, "*batch"] | None = None
    min_p: Float[Array, "*batch"] | None = None
    banned_tokens: Int[Array, "*batch max_banned_tokens"] | None = None
    repetition_penalty: Float[Array, "*batch"] | None = None
    presence_penalty: Float[Array, "*batch"] | None = None
    frequency_penalty: Float[Array, "*batch"] | None = None
    token_counts: Int[Array, "*batch vocabulary"] | None = None

    @classmethod
    def init(
        cls,
        temperature: float | None = None,
        top_k: int | None = None,
        top_p: float | None = None,
        min_p: float | None = None,
        banned_tokens: Iterable[int] | None = None,
        repetition_penalty: float | None = None,
        presence_penalty: float | None = None,
        frequency_penalty: float | None = None,
    ) -> "SamplingPolicy":
        banned_tokens = () if banned_tokens is None else tuple(banned_tokens)
        repetition_penalty = 1.0 if repetition_penalty is None else repetition_penalty
        presence_penalty = 0.0 if presence_penalty is None else presence_penalty
        frequency_penalty = 0.0 if frequency_penalty is None else frequency_penalty
        if repetition_penalty is not None and repetition_penalty <= 0.0:
            raise ValueError("repetition_penalty must be positive.")
        return cls(
            temperature=(
                None if temperature is None or temperature == 1.0 else jnp.asarray(temperature, dtype=jnp.float32)
            ),
            top_k=None if top_k is None or top_k <= 0 else jnp.asarray(top_k, dtype=jnp.int32),
            top_k_limit=None if top_k is None or top_k <= 0 else int(top_k),
            top_p=None if top_p is None or top_p >= 1.0 else jnp.asarray(top_p, dtype=jnp.float32),
            min_p=None if min_p is None or min_p <= 0.0 else jnp.asarray(min_p, dtype=jnp.float32),
            banned_tokens=(
                None if not banned_tokens else jnp.asarray(_pad_banned_tokens(banned_tokens), dtype=jnp.int32)
            ),
            repetition_penalty=(
                None if repetition_penalty == 1.0 else jnp.asarray(repetition_penalty, dtype=jnp.float32)
            ),
            presence_penalty=None if presence_penalty == 0.0 else jnp.asarray(presence_penalty, dtype=jnp.float32),
            frequency_penalty=None if frequency_penalty == 0.0 else jnp.asarray(frequency_penalty, dtype=jnp.float32),
            token_counts=None,
        )

    @classmethod
    def init_batch(
        cls,
        temperature: Iterable[float | None] | None = None,
        top_k: Iterable[int | None] | None = None,
        top_p: Iterable[float | None] | None = None,
        min_p: Iterable[float | None] | None = None,
        banned_tokens: Iterable[Iterable[int] | None] | None = None,
        repetition_penalty: Iterable[float | None] | None = None,
        presence_penalty: Iterable[float | None] | None = None,
        frequency_penalty: Iterable[float | None] | None = None,
    ) -> "SamplingPolicy":
        padded_banned_tokens = (
            None
            if banned_tokens is None
            else tuple(_pad_banned_tokens(()) if row is None else _pad_banned_tokens(row) for row in banned_tokens)
        )
        temperature = canonicalize(temperature, default=1.0, dtype=jnp.float32)
        top_k = canonicalize(top_k, default=0, dtype=jnp.int32)
        top_p = canonicalize(top_p, default=1.0, dtype=jnp.float32)
        min_p = canonicalize(min_p, default=0.0, dtype=jnp.float32)
        banned_tokens_array = (
            None
            if padded_banned_tokens is None or all(row == _pad_banned_tokens(()) for row in padded_banned_tokens)
            else jnp.asarray(padded_banned_tokens, dtype=jnp.int32)
        )
        repetition_penalty = canonicalize(repetition_penalty, default=1.0, dtype=jnp.float32)
        presence_penalty = canonicalize(presence_penalty, default=0.0, dtype=jnp.float32)
        frequency_penalty = canonicalize(frequency_penalty, default=0.0, dtype=jnp.float32)
        _raise_if_different_batch_sizes(
            *jax.tree.leaves(
                (
                    temperature,
                    top_k,
                    top_p,
                    min_p,
                    banned_tokens_array,
                    repetition_penalty,
                    presence_penalty,
                    frequency_penalty,
                ),
            ),
        )
        return cls(
            temperature=temperature,
            top_k=top_k,
            top_k_limit=None,
            top_p=top_p,
            min_p=min_p,
            banned_tokens=banned_tokens_array,
            repetition_penalty=repetition_penalty,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            token_counts=None,
        )

    @property
    def has_count_penalties(self) -> bool:
        return (
            self.repetition_penalty is not None
            or self.presence_penalty is not None
            or self.frequency_penalty is not None
        )

    def with_prompt_token_counts(
        self,
        prompt_token_ids: Int[Array, " tokens"],
        prompt_length: Int[Array, ""],
        vocabulary_size: int,
    ) -> "SamplingPolicy":
        token_ids = jnp.clip(prompt_token_ids, 0, vocabulary_size - 1)
        token_mask = (jnp.arange(prompt_token_ids.shape[0], dtype=jnp.int32) < prompt_length) & (
            (prompt_token_ids >= 0) & (prompt_token_ids < vocabulary_size)
        )
        token_counts = jnp.zeros(vocabulary_size, dtype=jnp.int32).at[token_ids].add(token_mask.astype(jnp.int32))
        return replace(self, token_counts=token_counts)

    def with_empty_token_counts(self, vocabulary_size: int) -> "SamplingPolicy":
        return replace(self, token_counts=jnp.zeros(vocabulary_size, dtype=jnp.int32))

    def with_next_token_count(
        self,
        token_id: Int[Array, ""],
        should_count: Bool[Array, ""] | bool = True,
    ) -> "SamplingPolicy":
        if self.token_counts is None:
            return self
        in_vocabulary = (token_id >= 0) & (token_id < self.token_counts.shape[0])
        token_id = jnp.clip(token_id, 0, self.token_counts.shape[0] - 1)
        count = (jnp.asarray(should_count) & in_vocabulary).astype(jnp.int32)
        return replace(self, token_counts=self.token_counts.at[token_id].add(count))

    def with_token_counts(
        self,
        token_counts: Int[Array, "... vocabulary"] | None,
    ) -> "SamplingPolicy":
        return replace(self, token_counts=token_counts)

    def token_counts_with_delta(
        self,
        base_token_counts: Int[Array, " vocabulary"],
        token_ids: Int[Array, " tokens"],
        token_mask: Bool[Array, " tokens"],
    ) -> Int[Array, " vocabulary"]:
        vocabulary_size = base_token_counts.shape[0]
        in_vocabulary = (token_ids >= 0) & (token_ids < vocabulary_size)
        safe_token_ids = jnp.clip(token_ids, 0, vocabulary_size - 1)
        updates = (token_mask & in_vocabulary).astype(jnp.int32)
        return base_token_counts.at[safe_token_ids].add(updates)

    def process_logits(
        self,
        logits: Float[Array, " vocabulary"],
    ) -> Float[Array, " vocabulary"]:
        self._raise_if_batched()
        logits = self._apply_banned_tokens(logits.astype(jnp.float32))
        logits = self._apply_count_penalties(logits)
        logits = self._apply_temperature(logits)
        logits = self._apply_top_k(logits)
        logits = self._apply_top_p(logits)
        return self._apply_min_p(logits)

    def broadcast(self, batch_size: int) -> "SamplingPolicy":
        def broadcast_leaf(leaf: object) -> object:
            if isinstance(leaf, jax.Array):
                return jnp.broadcast_to(leaf, (batch_size, *leaf.shape))
            return leaf

        return jax.tree.map(broadcast_leaf, self)

    def reshape(
        self,
        old_leading_shape: tuple[int, ...],
        new_leading_shape: tuple[int, ...],
    ) -> "SamplingPolicy":
        return jax.tree.map(
            lambda value: (
                value.reshape((*new_leading_shape, *value.shape[len(old_leading_shape) :]))
                if eqx.is_array(value)
                else value
            ),
            self,
        )

    def sample(
        self,
        logits: Float[Array, " vocabulary"],
        key: Key[Array, ""],
        should_count: Bool[Array, ""] | bool = True,
        update_token_counts: bool = True,
    ) -> tuple[Float[Array, " vocabulary"], Int[Array, ""], "SamplingPolicy"]:
        processed_logits = self.process_logits(logits.astype(jnp.float32))
        token_id = jax.random.categorical(key, processed_logits).astype(jnp.int32)
        next_policy = self
        if update_token_counts:
            next_policy = next_policy.with_next_token_count(token_id, should_count)
        return processed_logits, token_id, next_policy

    def sample_top_k(
        self,
        logits: Float[Array, " vocabulary"],
        key: Key[Array, ""],
        k: int,
    ) -> Int[Array, " k"]:
        processed_logits = self.process_logits(logits.astype(jnp.float32))
        return jax.random.categorical(key, processed_logits, shape=(k,), replace=False).astype(jnp.int32)

    def __call__(self, logits: Float[Array, " vocabulary"], *, keychain: Keychain) -> Int[Array, ""]:
        self._raise_if_batched()
        return jax.random.categorical(keychain.vmapped_keys, self.process_logits(logits))

    def _raise_if_batched(self) -> None:
        scalar_fields: tuple[SamplingLeaf | None, ...] = (
            self.temperature,
            self.top_k,
            self.top_p,
            self.min_p,
            self.repetition_penalty,
            self.presence_penalty,
            self.frequency_penalty,
        )
        vector_fields: tuple[SamplingLeaf | None, ...] = (self.banned_tokens, self.token_counts)
        if any(field is not None and field.ndim != 0 for field in scalar_fields) or any(
            field is not None and field.ndim != 1 for field in vector_fields
        ):
            raise ValueError(
                "Attempted to call a method on a batched version of SamplingPolicy. Use vmap instead.",
            )

    def _apply_banned_tokens(self, logits: Float[Array, " vocabulary"]) -> Float[Array, " vocabulary"]:
        if self.banned_tokens is None:
            return logits
        (vocabulary_size,) = logits.shape
        vocabulary_indices = jnp.arange(vocabulary_size)
        banned_token_mask = jnp.any(
            self.banned_tokens[:, None] == vocabulary_indices,
            axis=0,
        )
        return jnp.where(banned_token_mask, -jnp.inf, logits)

    def _apply_count_penalties(
        self,
        logits: Float[Array, " vocabulary"],
    ) -> Float[Array, " vocabulary"]:
        if not self.has_count_penalties or self.token_counts is None:
            return logits

        if self.repetition_penalty is not None:
            logits = apply_repetition_penalty(logits, self.token_counts > 0, self.repetition_penalty)

        if self.presence_penalty is not None:
            logits = jnp.where(self.token_counts > 0, logits - self.presence_penalty, logits)

        if self.frequency_penalty is not None:
            logits = logits - self.frequency_penalty * self.token_counts.astype(logits.dtype)

        return logits

    def _apply_temperature(self, logits: Float[Array, " vocabulary"]) -> Float[Array, " vocabulary"]:
        if self.temperature is None:
            return logits
        (vocabulary_size,) = logits.shape
        best_token = jnp.argmax(logits, axis=-1)
        greedy_logits = jnp.where(jnp.arange(vocabulary_size) == best_token, 1.0, -jnp.inf)
        return jnp.where(
            self.temperature == 0.0,
            greedy_logits,
            logits / jnp.where(self.temperature == 0.0, 1.0, self.temperature),
        )

    def _apply_top_k(self, logits: Float[Array, " vocabulary"]) -> Float[Array, " vocabulary"]:
        if self.top_k is None:
            return logits
        (vocabulary_size,) = logits.shape
        effective_top_k = jnp.clip(self.top_k, 1, vocabulary_size)
        if self.top_k_limit is None:
            sorted_logits = jnp.sort(logits, axis=-1, descending=True)
            min_logit = sorted_logits[effective_top_k - 1]
        else:
            top_values, _top_indices = jax.lax.top_k(logits, min(self.top_k_limit, vocabulary_size))
            min_logit = top_values[effective_top_k - 1]
        filtered_logits = jnp.where(logits >= min_logit, logits, -jnp.inf)
        return jnp.where(self.top_k > 0, filtered_logits, logits)

    def _apply_top_p(self, logits: Float[Array, " vocabulary"]) -> Float[Array, " vocabulary"]:
        if self.top_p is None:
            return logits
        if self.top_k_limit is not None:
            return self._apply_top_p_inside_static_top_k(logits)
        sorted_indices = jnp.argsort(logits, axis=-1, descending=True)
        sorted_logits = jnp.take_along_axis(logits, sorted_indices, axis=-1)
        cumulative_probs = jnp.cumsum(jax.nn.softmax(sorted_logits, axis=-1), axis=-1)

        to_remove_sorted = cumulative_probs > self.top_p
        to_remove_sorted = jnp.roll(to_remove_sorted, shift=1, axis=-1)
        to_remove_sorted = to_remove_sorted.at[0].set(False)

        unsort_indices = jnp.argsort(sorted_indices, axis=-1)
        to_remove_unsorted = jnp.take_along_axis(to_remove_sorted, unsort_indices, axis=-1)

        return jnp.where(to_remove_unsorted, -jnp.inf, logits)

    def _apply_top_p_inside_static_top_k(self, logits: Float[Array, " vocabulary"]) -> Float[Array, " vocabulary"]:
        assert self.top_k_limit is not None
        assert self.top_p is not None
        (vocabulary_size,) = logits.shape
        top_values, top_indices = jax.lax.top_k(logits, min(self.top_k_limit, vocabulary_size))
        cumulative_probs = jnp.cumsum(jax.nn.softmax(top_values, axis=-1), axis=-1)
        to_remove = cumulative_probs > self.top_p
        to_remove = jnp.roll(to_remove, shift=1, axis=-1)
        to_remove = to_remove.at[0].set(False)
        filtered_top_values = jnp.where(to_remove, -jnp.inf, top_values)
        return jnp.full_like(logits, -jnp.inf).at[top_indices].set(filtered_top_values)

    def _apply_min_p(self, logits: Float[Array, " vocabulary"]) -> Float[Array, " vocabulary"]:
        if self.min_p is None:
            return logits
        max_logit = jnp.max(logits)
        logit_cutoff = max_logit + jnp.log(self.min_p)
        filtered_logits = jnp.where(logits >= logit_cutoff, logits, -jnp.inf)
        return jnp.where(self.min_p == 0.0, logits, filtered_logits)


def _raise_if_different_batch_sizes(*arrays: SamplingLeaf) -> None:
    if not arrays:
        return
    first_size = arrays[0].shape[0]
    if any(array.shape[0] != first_size for array in arrays):
        raise ValueError("init_batch iterable arguments must have the same length.")


def apply_repetition_penalty(
    logits: Float[Array, " vocabulary"],
    seen_token_mask: Bool[Array, " vocabulary"],
    repetition_penalty: Float[Array, ""],
) -> Float[Array, " vocabulary"]:
    penalized_logits = jnp.where(logits > 0, logits / repetition_penalty, logits * repetition_penalty)
    return jnp.where(seen_token_mask, penalized_logits, logits)


def canonicalize[T](
    values: Iterable[T | None] | None,
    *,
    default: T,
    dtype: DTypeLike,
) -> SamplingLeaf | None:
    if values is None:
        return None
    values = tuple(default if value is None else value for value in values)
    if all(value == default for value in values):
        return None
    return jnp.asarray(values, dtype=dtype)


def _pad_banned_tokens(banned_tokens: Iterable[int]) -> tuple[int, ...]:
    tokens = tuple(banned_tokens)
    if len(tokens) > _MAX_BANNED_TOKENS:
        raise ValueError(f"At most {_MAX_BANNED_TOKENS} banned tokens are supported.")
    if any(token < 0 for token in tokens):
        raise ValueError(f"Banned tokens must be non-negative token ids. {_SENTINEL} is reserved as a sentinel.")
    return tokens + (_SENTINEL,) * (_MAX_BANNED_TOKENS - len(tokens))
