from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Callable, Iterable, Sequence

from .common import ContractViolation
from .schema import parse_wrapped_findings


PREFIX = '<answer>{"findings":['
SUFFIX = '}</answer>'


@dataclass(frozen=True)
class GrammarState:
    mode: str = "prefix"
    literal_pos: int = 0
    item_count: int = 0
    has_content: bool = False

    @property
    def complete(self) -> bool:
        return self.mode == "complete"

    def feed_char(self, character: str) -> "GrammarState" | None:
        if len(character) != 1:
            raise ValueError("feed_char expects one character")
        if self.mode == "prefix":
            if self.literal_pos >= len(PREFIX) or character != PREFIX[self.literal_pos]:
                return None
            next_position = self.literal_pos + 1
            return GrammarState("array", 0, 0, False) if next_position == len(PREFIX) else GrammarState("prefix", next_position, 0, False)
        if self.mode == "array":
            if character == "]":
                return GrammarState("suffix", 0, self.item_count, False)
            if character == '"' and self.item_count < 5:
                return GrammarState("string", 0, self.item_count + 1, False)
            return None
        if self.mode == "string":
            if character == '"':
                return GrammarState("after_string", 0, self.item_count, False) if self.has_content else None
            if 32 <= ord(character) <= 126 and character not in {'\\', '<', '>'}:
                return GrammarState("string", 0, self.item_count, True)
            return None
        if self.mode == "after_string":
            if character == "]":
                return GrammarState("suffix", 0, self.item_count, False)
            if character == "," and self.item_count < 5:
                return GrammarState("after_comma", 0, self.item_count, False)
            return None
        if self.mode == "after_comma":
            return GrammarState("string", 0, self.item_count + 1, False) if character == '"' else None
        if self.mode == "suffix":
            if self.literal_pos >= len(SUFFIX) or character != SUFFIX[self.literal_pos]:
                return None
            next_position = self.literal_pos + 1
            return GrammarState("complete", 0, self.item_count, False) if next_position == len(SUFFIX) else GrammarState("suffix", next_position, self.item_count, False)
        return None

    def feed_text(self, text: str) -> "GrammarState" | None:
        state: GrammarState | None = self
        for character in text:
            if state is None:
                return None
            state = state.feed_char(character)
        return state


def grammar_state(text: str) -> GrammarState:
    state = GrammarState().feed_text(text)
    if state is None:
        raise ContractViolation("Text is not a prefix of the constrained findings language")
    return state


class FindingsPrefixConstraint:
    """Transformers-compatible prefix constraint for canonical wrapped findings JSON.

    The grammar only permits printable ASCII finding strings and 0-5 non-empty
    items. It constrains generation itself; it never repairs or rewrites output.
    """

    _token_text_cache: dict[tuple[int, int], tuple[str | None, ...]] = {}
    _allowed_state_cache: dict[tuple[int, int, GrammarState], tuple[int, ...]] = {}

    def __init__(self, tokenizer: Any, prompt_lengths: Sequence[int]) -> None:
        self.tokenizer = tokenizer
        self.prompt_lengths = tuple(int(value) for value in prompt_lengths)
        self.eos_token_id = getattr(tokenizer, "eos_token_id", None)
        if self.eos_token_id is None:
            raise ContractViolation("Tokenizer must expose eos_token_id")
        vocabulary_size = len(tokenizer)
        self._cache_key = (id(tokenizer), vocabulary_size)
        cached = self._token_text_cache.get(self._cache_key)
        if cached is None:
            special_ids = set(getattr(tokenizer, "all_special_ids", []))
            token_text: list[str | None] = []
            for token_id in range(vocabulary_size):
                if token_id in special_ids:
                    token_text.append(None)
                    continue
                text = tokenizer.decode([token_id], skip_special_tokens=False, clean_up_tokenization_spaces=False)
                if not text or any(ord(character) > 126 for character in text):
                    token_text.append(None)
                else:
                    token_text.append(text)
            cached = tuple(token_text)
            self._token_text_cache[self._cache_key] = cached
        self._token_text = cached

    @lru_cache(maxsize=256)
    def _allowed_for_state(self, state: GrammarState) -> tuple[int, ...]:
        if state.complete:
            return (int(self.eos_token_id),)
        shared_key = (*self._cache_key, state)
        shared = self._allowed_state_cache.get(shared_key)
        if shared is not None:
            return shared
        allowed = []
        for token_id, text in enumerate(self._token_text):
            if text is not None and state.feed_text(text) is not None:
                allowed.append(token_id)
        if not allowed:
            raise ContractViolation(f"Tokenizer has no token continuing grammar state {state}")
        result = tuple(allowed)
        self._allowed_state_cache[shared_key] = result
        return result

    def __call__(self, batch_id: int, input_ids: Any) -> list[int]:
        prompt_length = self.prompt_lengths[batch_id]
        ids = input_ids.tolist() if hasattr(input_ids, "tolist") else list(input_ids)
        generated_ids = ids[prompt_length:]
        text = self.tokenizer.decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        state = grammar_state(text)
        return list(self._allowed_for_state(state))


def constrained_generation_kwargs(tokenizer: Any, prompt_lengths: Sequence[int]) -> dict[str, Any]:
    constraint = FindingsPrefixConstraint(tokenizer, prompt_lengths)
    return {
        "do_sample": False,
        "prefix_allowed_tokens_fn": constraint,
        "use_cache": True,
    }


def evaluate_native_and_deployed(native_raw: str, deployed_raw: str) -> dict[str, Any]:
    native = parse_wrapped_findings(native_raw)
    deployed = parse_wrapped_findings(deployed_raw)
    return {
        "native_schema_valid": native.valid,
        "native_schema_error": native.error,
        "deployed_schema_valid": deployed.valid,
        "deployed_schema_error": deployed.error,
        "metrics_separated": True,
    }
