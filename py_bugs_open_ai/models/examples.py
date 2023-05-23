from typing import List

from py_bugs_open_ai.models.base import MyBaseModel


class Example(MyBaseModel):
    code: str
    response: str


class ExamplesFile(MyBaseModel):
    examples: List[Example]
