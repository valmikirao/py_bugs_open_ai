import json
from dataclasses import dataclass, field
from typing import List, Iterable, NamedTuple

import openai
from openai.error import RateLimitError
from tenacity import retry, retry_if_exception, wait_exponential

from py_bugs_open_ai.constants import DEFAULT_MODEL, EMBEDDINGS_REQUEST_CHUNK_SIZE, DEFAULT_EMBEDDINGS_MODEL
from py_bugs_open_ai.models.base import CacheProtocol
from py_bugs_open_ai.models.open_ai import OpenAiResponse, Message, EmbeddingResponse

rate_limit_retry = retry(
    retry=retry_if_exception(lambda e: isinstance(e, RateLimitError)),
    wait=wait_exponential(multiplier=2, exp_base=10, max=5 * 60 * 60)
)


class EmbeddingItem(NamedTuple):
    text: str
    embeddings: List[float]


@dataclass
class OpenAiClient:
    api_key: str
    model: str = DEFAULT_MODEL
    embedding_model: str = DEFAULT_EMBEDDINGS_MODEL
    query_cache: CacheProtocol[str, str] = field(default_factory=lambda: {})
    embeddings_cache: CacheProtocol[str, List[float]] = field(default_factory=lambda: {})

    @rate_limit_retry
    def query_messages(self, messages: List[Message], refresh_cache: bool = False) -> str:
        message_dicts = [m.full_dict() for m in messages]
        cache_key = json.dumps(message_dicts, sort_keys=True)
        if refresh_cache or cache_key not in self.query_cache:
            # this except secure
            openai.api_key = self.api_key

            # response = openai.Completion.create(
            response_raw = openai.ChatCompletion.create(
                model=self.model,
                messages=message_dicts,
                temperature=0,
                stream=False
            )
            response = OpenAiResponse.parse_obj(response_raw)

            # can you create a pydantic model to handle this response?
            return_text = ''
            choices = response.choices
            choices = sorted(choices, key=lambda c: c.index)
            for choice in choices:
                return_text += choice.message.content

            self.query_cache[cache_key] = return_text

        return self.query_cache[cache_key]

    @rate_limit_retry
    def _get_embeddings(self, input_: List[str]) -> EmbeddingResponse:
        openai.api_key = self.api_key
        response_raw = openai.Embedding.create(model=self.embedding_model, input=input_)
        return EmbeddingResponse.parse_obj(response_raw)

    def get_embeddings(self, texts: Iterable[str], refresh_cache: bool = False,
                       embeddings_request_chunk_size: int = EMBEDDINGS_REQUEST_CHUNK_SIZE) \
            -> Iterable[EmbeddingItem]:
        chunk_to_get_embeddings: List[str] = []

        def _get_embeddings_for_chunk(text_chunk: List[str]) -> Iterable[EmbeddingItem]:
            embedding_response = self._get_embeddings(text_chunk)
            for response_item, text_ in zip(embedding_response.data, text_chunk):
                self.embeddings_cache[text_] = response_item.embedding
                yield EmbeddingItem(text_, response_item.embedding)

        for text in texts:
            if not refresh_cache and text in self.embeddings_cache:
                yield EmbeddingItem(
                    text=text,
                    embeddings=self.embeddings_cache[text]
                )
            else:
                chunk_to_get_embeddings.append(text)
                if len(chunk_to_get_embeddings) >= embeddings_request_chunk_size:
                    yield from _get_embeddings_for_chunk(chunk_to_get_embeddings)
                    chunk_to_get_embeddings = []
        if len(chunk_to_get_embeddings) > 0:
            yield from _get_embeddings_for_chunk(chunk_to_get_embeddings)
