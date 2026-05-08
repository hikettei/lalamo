from collections.abc import Callable, Iterable
from itertools import batched, chain
from typing import NamedTuple

import jax

from lalamo.data.lalamo_completions import LalamoCompletion
from lalamo.data.utils import get_prefixes_ending_in_user_message
from lalamo.message_processor import Message
from lalamo.models import LanguageModel
from lalamo.models.batch_scheduler import ContinuousBatchScheduler
from lalamo.models.common import InferenceConfig
from lalamo.modules import MLPForwardPassConfig


class CollectTracesEvent(NamedTuple):
    sequences_processed: int
    total_sequences: int | None
    tokens_generated: int


def inference_collect_traces(
    model: LanguageModel,
    conversations: Iterable[Iterable[Message]],
    batch_size: int = 1,
    max_input_length: int = 1024,
    max_output_length: int = 4096,
    tokens_to_generate: int | None = None,
    progress_callback: Callable[[CollectTracesEvent], None] | None = None,
) -> Iterable[LalamoCompletion]:
    prefixes = chain.from_iterable(map(get_prefixes_ending_in_user_message, conversations))
    tokenized_prefixes = map(model.message_processor.tokenize_request, prefixes)
    filtered_prefixes = filter(lambda conv: len(conv) <= max_input_length, tokenized_prefixes)

    config = InferenceConfig(
        max_output_length=max_output_length,
        padded_length=max_input_length,
        batch_size=batch_size,
    )
    tokens_generated = 0
    sequences_processed = 0
    key = jax.random.key(0)
    scheduler = ContinuousBatchScheduler(model=model)
    chunk_size = batch_size * 256

    for prefix_chunk in batched(filtered_prefixes, chunk_size):
        keys = jax.random.split(key, len(prefix_chunk) + 1)
        key = keys[0]
        generated_batch = scheduler.generate_tokens_many(
            prefix_chunk,
            generation_config=model.config.generation_config,
            inference_config=config,
            forward_pass_config=MLPForwardPassConfig(),
            keys=keys[1:],
        )
        for prefix_idx, generated in generated_batch:
            prefix_token_ids = prefix_chunk[prefix_idx]
            token_ids = generated.token_ids.tolist()
            seqlen = next(
                (i + 1 for i, t in enumerate(token_ids) if t in model.stop_token_ids),
                len(token_ids),
            )

            if tokens_to_generate is not None:
                seqlen = min(seqlen, tokens_to_generate - tokens_generated)

            tokens_generated += seqlen
            sequences_processed += 1
            token_ids = token_ids[:seqlen]

            yield LalamoCompletion(
                prefix_token_ids=prefix_token_ids,
                completion_token_ids=token_ids,
            )

            if progress_callback is not None:
                progress_callback(CollectTracesEvent(sequences_processed, None, tokens_generated))

            if tokens_to_generate is not None and tokens_generated >= tokens_to_generate:
                return
