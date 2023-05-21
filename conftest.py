import os
from typing import List

import pytest

from py_bugs_open_ai.models.examples import ExamplesFile, Example


class ExampleWithEmbeds(Example):
    embeddings: List[float]


@pytest.fixture
def examples_file() -> ExamplesFile:
    return ExamplesFile(
        examples=[
            ExampleWithEmbeds(code='print(x)', response="ERROR: NameError: name 'x' is not defined",
                              embeddings=[0.0, 1.0]),
            ExampleWithEmbeds(code='a = "hello"\nb = 42\nprint(a + b)\n',
                              response='ERROR: TypeError: can only concatenate str (not "int") to str',
                              embeddings=[0.25 ** .5, .75 ** .5]),
            ExampleWithEmbeds(code='my_list = [1, 2, 3]\nprint(my_list[3])\n',
                              response='ERROR: list index out of range', embeddings=[.5 ** .5, .5 ** .5]),
            ExampleWithEmbeds(code='my_dict = {"another_key": "another_value"}\nprint(my_dict["my_key"])\n',
                              response="KeyError: 'my_key'", embeddings=[.75 ** .5, .25 ** .5]),
            ExampleWithEmbeds(code='x = 5\ny = 0\nprint(x / y)\n',
                              response='ZeroDivisionError: division by zero', embeddings=[1.0, 0])
        ]
    )


@pytest.fixture
def base_dir() -> str:
    return os.path.dirname(__file__)


@pytest.fixture
def base_chunker_param_dir(base_dir: str) -> str:
    return os.path.join(base_dir, 'tests', 'resources', 'test-chunker-params')
