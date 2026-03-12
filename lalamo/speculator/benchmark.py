from abc import abstractmethod
from collections.abc import Callable, Iterable, Sequence
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np
from jaxtyping import Float, Int, Array

from lalamo.models import LanguageModel
from lalamo.models.language_model import DecodingState, _COMPILED_PROMPT_LENGTHS
from lalamo.modules import ForwardPassMode
from lalamo.speculator.common import Speculator


def _logsumexp(x: np.ndarray) -> float:
    x_max = np.max(x)
    return float(x_max + np.log(np.sum(np.exp(x - x_max))))


class AcceptanceMetric:
    @abstractmethod
    def update(self, target_logits: np.ndarray, draft_probs: dict[int, float]) -> None:
        raise NotImplementedError

    @abstractmethod
    def result(self) -> float:
        raise NotImplementedError


class RejectionSamplingMetric(AcceptanceMetric):
    def __init__(self) -> None:
        self._total_acceptance = 0.0
        self._count = 0

    def update(self, target_logits: np.ndarray, draft_probs: dict[int, float]) -> None:
        lse = _logsumexp(target_logits)
        draft_ids = list(draft_probs.keys())
        q_vals = np.array(list(draft_probs.values()))
        p_at_draft = np.exp(target_logits[draft_ids] - lse)
        self._total_acceptance += float(np.minimum(p_at_draft, q_vals).sum())
        self._count += 1

    def result(self) -> float:
        if self._count == 0:
            return 0.0
        return self._total_acceptance / self._count


class SupportMassMetric(AcceptanceMetric):
    def __init__(self) -> None:
        self._total_mass = 0.0
        self._count = 0

    def update(self, target_logits: np.ndarray, draft_probs: dict[int, float]) -> None:
        lse = _logsumexp(target_logits)
        draft_ids = list(draft_probs.keys())
        p_at_draft = np.exp(target_logits[draft_ids] - lse)
        self._total_mass += float(p_at_draft.sum())
        self._count += 1

    def result(self) -> float:
        if self._count == 0:
            return 0.0
        return self._total_mass / self._count

class GumbelCouplingMetric(AcceptanceMetric):
    def __init__(self, num_samples: int = 10000, seed: int = 42) -> None:
        self._total_acceptance = 0.0
        self._count = 0
        self._num_samples = num_samples
        self._rng = np.random.default_rng(seed)

    def update(self, target_logits: np.ndarray, draft_probs: dict[int, float]) -> None:
        if not draft_probs or all(v == 0.0 for v in draft_probs.values()):
            self._count += 1
            return

        lse = _logsumexp(target_logits)
        draft_ids = list(draft_probs.keys())
        q_topk = np.array(list(draft_probs.values()))
        p_topk = np.exp(target_logits[draft_ids] - lse)
        p_out = 1.0 - p_topk.sum()

        K = len(draft_ids)
        E = self._rng.exponential(1.0, size=(self._num_samples, K))
        U = self._rng.uniform(0.0, 1.0, size=self._num_samples)

        draft_winners = np.argmin(E / q_topk, axis=1)
        target_winners = np.argmin(E / p_topk, axis=1)
        T_S = E[np.arange(self._num_samples), target_winners] / p_topk[target_winners]

        tail_beats = U > np.exp(-p_out * T_S)
        accepted = (draft_winners == target_winners) & ~tail_beats

        self._total_acceptance += float(accepted.mean())
        self._count += 1

    def result(self) -> float:
        if self._count == 0:
            return 0.0
        return self._total_acceptance / self._count


class BenchmarkEvent(NamedTuple):
    benchmarked_sequences: int
    benchmarked_tokens: int


class BenchmarkResult(NamedTuple):
    total_tokens: int
    total_sequences: int

# Note(hikettei)
# The metrics need full target probabilities p_i (all vocabs), not just top-K.
# Two approaches:
# - Online: run the target model during benchmark to get full logits (slower, but exact)
# - Offline: store logsumexp during tracing, reconstruct full softmax from top-K + lse (faster)
# Currently using the online approach. but we should transfer to offline approach ig
def benchmark_speculator(
    model: LanguageModel,
    speculator: Speculator,
    prompts: Iterable[list[int]],
    metrics: Sequence[AcceptanceMetric],
    max_output_length: int = 1024,
    progress_callback: Callable[[BenchmarkEvent], None] | None = None,
) -> BenchmarkResult:
    total_tokens = 0
    seq_count = 0

    eos_token_ids = jnp.array(model.stop_token_ids, dtype=jnp.int32)

    for seq_count, prompt in enumerate(prompts, start=1):
        input_length = len(prompt)
        padded_input_length = min(
            length for length in _COMPILED_PROMPT_LENGTHS if length >= input_length
        )
        padded_token_ids = jnp.zeros((padded_input_length,), dtype=jnp.int32)
        padded_token_ids = padded_token_ids.at[:input_length].set(jnp.array(prompt, dtype=jnp.int32))

        prefill_results = model._prefill(
            padded_token_ids[None, :],
            padded_input_length + max_output_length,
            lengths_without_padding=jnp.array([input_length], dtype=jnp.int32),
        )

        state = DecodingState(
            prefill_results.last_token_logits,
            prefill_results.last_token_indices,
            prefill_results.state,
            jnp.array([False], dtype=jnp.bool),
        )

        context: list[int] = []

        for _ in range(max_output_length):
            logits = np.asarray(state.last_token_logits.astype(jnp.float32).squeeze(0))
            draft_probs = speculator.probs(context)

            for metric in metrics:
                metric.update(logits, draft_probs)

            total_tokens += 1
            next_token = int(np.argmax(logits))
            context.append(next_token)

            if jnp.any(jnp.array(next_token) == eos_token_ids):
                break

            next_token_indices = state.last_token_indices + 1
            decoder_outputs = model.model(
                jnp.array(next_token, dtype=jnp.int32).reshape(1, 1),
                next_token_indices.reshape(1, 1),
                state.state,
                return_updated_state=True,
                forward_pass_mode=ForwardPassMode.SINGLE_TOKEN,
            )
            assert decoder_outputs.updated_state is not None
            state = DecodingState(
                decoder_outputs.logits.squeeze(1),
                next_token_indices,
                decoder_outputs.updated_state,
                state.stop_flags,
            )

        if progress_callback is not None:
            progress_callback(BenchmarkEvent(seq_count, total_tokens))

    return BenchmarkResult(
        total_tokens=total_tokens,
        total_sequences=seq_count if total_tokens > 0 else 0,
    )
