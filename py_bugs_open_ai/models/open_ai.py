from enum import Enum
from typing import List
from py_bugs_open_ai.models.base import MyBaseModel


class Role(Enum):
    user = 'user'
    system = 'system'
    agent = 'assistant'


class Message(MyBaseModel):
    role: Role
    content: str

    def __init__(self, role: Role, content: str):
        super().__init__(role=role, content=content)


class Choice(MyBaseModel):
    index: int
    message: Message


class OpenAiResponse(MyBaseModel):
    choices: List[Choice]


class Embedding(MyBaseModel):
    embedding: List[float]


class EmbeddingResponse(MyBaseModel):
    data: List[Embedding]
