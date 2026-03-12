from abc import abstractmethod
from collections.abc import Callable, Iterable, Sequence
from typing import NamedTuple

from lalamo.data.lalamo_completions import LalamoCompletion
from lalamo.speculator.common import Speculator
from lalamo.speculator.ngram import softmax


class BenchmarkEvent(NamedTuple):
    benchmarked_sequences: int
    benchmarked_tokens: int

class BenchmarkResult(NamedTuple):
    total_tokens: int
    total_sequences: int


class AcceptanceMetric:
    @abstractmethod
    def update(self, target_logits: dict[int, float], draft_probs: dict[int, float]) -> None:
        raise NotImplementedError

    @abstractmethod
    def result(self) -> float:
        raise NotImplementedError


class RejectionSamplingMetric(AcceptanceMetric):
    def __init__(self) -> None:
        self._total_acceptance = 0.0
        self._count = 0

    def update(self, target_logits: dict[int, float], draft_probs: dict[int, float]) -> None:
        p = dict(zip(target_logits.keys(), softmax(target_logits.values()), strict=True))
        all_keys = p.keys() | draft_probs.keys()
        self._total_acceptance += sum(min(p.get(k, 0.0), draft_probs.get(k, 0.0)) for k in all_keys)
        self._count += 1

    def result(self) -> float:
        if self._count == 0:
            return 0.0
        return self._total_acceptance / self._count

# note(hikettei):
# (plus, are the logits really needed???)
# α = P(argmax(log p_i + g_i) = argmax(log q_i + g_i))
# so p_i and q_i are independent.
# Can we compute acceptance rate in "closed-forms", for top-K draft tokens,
# and sampled by Gumbel, without using Monte Carlo?
# We do not want to compute "full-vocab Gumbel"
class GumbelCouplingMetric(AcceptanceMetric):
    def update(self, target_logits: dict[int, float], draft_probs: dict[int, float]) -> None:
        raise NotImplementedError

    def result(self) -> float:
        raise NotImplementedError

def benchmark_speculator(
    speculator: Speculator,
    traces: Iterable[LalamoCompletion],
    metrics: Sequence[AcceptanceMetric],
    progress_callback: Callable[[BenchmarkEvent], None] | None = None,
) -> BenchmarkResult:
    total_tokens = 0
    seq_count = 0

    for seq_count, trace in enumerate(traces, start=1):
        for t, logits in enumerate(trace.completion_token_logits):
            context = trace.completion_token_ids[:t]
            draft_probs = speculator.probs(context)

            for metric in metrics:
                metric.update(logits, draft_probs)

            total_tokens += 1

        if progress_callback is not None:
            progress_callback(BenchmarkEvent(seq_count, total_tokens))

    return BenchmarkResult(
        total_tokens=total_tokens,
        total_sequences=seq_count if total_tokens > 0 else 0,
    )
