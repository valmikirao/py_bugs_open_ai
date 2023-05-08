import json
from dataclasses import dataclass, field
from typing import List

import openai
from openai.error import RateLimitError
from tenacity import retry, retry_if_exception, wait_exponential

from py_bugs_open_ai.constants import DEFAULT_MODEL
from py_bugs_open_ai.models.base import CacheProtocol
from py_bugs_open_ai.models.open_ai import OpenAiResponse, Message


@dataclass
class OpenAiClient:
    api_key: str
    model: str = DEFAULT_MODEL
    cache: CacheProtocol[str, str] = field(default_factory=lambda: {})

    @retry(
        retry=retry_if_exception(lambda e: isinstance(e, RateLimitError)),
        wait=wait_exponential(multiplier=2, exp_base=10, max=5 * 60 * 60)
    )
    def query_messages(self, messages: List[Message]) -> str:
        message_dicts = [m.full_dict() for m in messages]
        cache_key = json.dumps(message_dicts, sort_keys=True)
        if cache_key not in self.cache:
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

            self.cache[cache_key] = return_text

        return self.cache[cache_key]
