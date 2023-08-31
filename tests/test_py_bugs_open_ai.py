#!/usr/bin/env python

"""Tests for `py_bugs_open_ai` package."""
import itertools
import os
import re
from random import Random
from typing import List, Any, NamedTuple, Set, cast, Iterable, Optional
from unittest.mock import create_autospec

from conftest import ExampleWithEmbeds
from py_bugs_open_ai.models.examples import ExamplesFile
from py_bugs_open_ai.models.open_ai import Message, Role
from py_bugs_open_ai.open_ai_client import OpenAiClient, EmbeddingItem

import pytest
import tiktoken

from pydantic import BaseModel

import ast
from py_bugs_open_ai.constants import DEFAULT_MODEL, DEFAULT_MAX_TOKENS_TO_SEND, DEFAULT_MAX_CHUNK_SIZE
from py_bugs_open_ai.py_bugs_open_ai import CodeChunker, CodeChunk, QueryConstructor, CHUNK_PARSE_MESSAGE


@pytest.mark.parametrize("total_token_count,max_chunk_size,expected", [
    (100, 10, 10),
    (100, 50, 50),
    (100, 49, 33),
    (100, 100, 100),
    (100, 200, 100),
    (100, 1, 1),
    (100, 99, 50),
    (100, 98, 50),
    (100, 97, 50),
])
def test_get_goal_min_size(total_token_count, max_chunk_size, expected):
    assert CodeChunker.get_goal_min_size(total_token_count, max_chunk_size) == expected


class ExpectedChunk(BaseModel):
    code: Optional[str] = None
    token_count: Optional[int] = None
    error: Optional[str] = None
    warning: Optional[str] = None

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, CodeChunk):
            if self.code is not None:
                if self.code != other.code:
                    return False
            if self.token_count is not None:
                if self.token_count != other.token_count:
                    return False
            return self.error == other.error and self.warning == other.warning
        else:
            return super().__eq__(other)


class ChunkerParametrize(NamedTuple):
    file: str = 'test-1.py'
    max_chunk_size: int = 50
    abs_max_chunk_size: int = -1
    strict_chunk_size: bool = False
    expected_token_counts: List[int] = []
    expected_errors_i: Set[int] = set()
    expected_warnings_i: Set[int] = set()

    @classmethod
    def parametrize(cls):
        return pytest.mark.parametrize(
            'file,max_chunk_size,abs_max_chunk_size,strict_chunk_size,expected_token_counts,expected_errors_i,'
            'expected_warnings_i', [
                cls(expected_token_counts=[37, 30, 13, 37, 10]),
                cls(max_chunk_size=20, expected_token_counts=[16, 16, 11, 14, 13, 18, 18, 10]),
                cls(file='test-long-list.py', expected_token_counts=[284], expected_warnings_i={0}),
                cls(file='test-long-list.py', expected_token_counts=[284], expected_errors_i={0},
                    strict_chunk_size=True),
                cls(file='test-long-list.py', expected_token_counts=[284], abs_max_chunk_size=300),
                cls(file='test-multi-peer-groups.py', expected_token_counts=[27, 31, 44, 29, 43, 44]),
                cls(file='test-multi-peer-groups.py', max_chunk_size=200, expected_token_counts=[104, 118]),
                cls(file='test-multi-peer-groups.py', max_chunk_size=20,
                    expected_token_counts=[6, 7, 14, 11, 18, 11, 9, 19, 3, 6, 7, 16, 11, 9, 18, 3, 11, 9, 19, 3])
            ]
        )


@ChunkerParametrize.parametrize()
def test_chunker(file: str, max_chunk_size: int, abs_max_chunk_size: int, strict_chunk_size: bool,
                 expected_token_counts: List[int], expected_errors_i: Set[int], expected_warnings_i: Set[int],
                 base_chunker_param_dir: str):
    file_ = os.path.join(base_chunker_param_dir, file)
    with open(file_, 'r') as f:
        code = f.read()

    chunker = CodeChunker(
        code=code,
        file=file_,
        max_chunk_size=max_chunk_size,
        abs_max_chunk_size=abs_max_chunk_size,
        strict_chunk_size=strict_chunk_size
    )
    chunks = list(chunker.get_chunks())

    try:
        assert [c.token_count for c in chunks] == expected_token_counts
    except AssertionError:
        for chunk, expected_token_count in itertools.zip_longest(chunks, expected_token_counts):
            token_count = chunk.token_count if chunk else None
            print(f'------ actual token count: {token_count}; expected token count {expected_token_count} ------')
            if chunk:
                print(chunk.code)
        raise
    assert set(i for i, c in enumerate(chunks) if c.error is not None) == expected_errors_i
    assert set(i for i, c in enumerate(chunks) if c.warning is not None) == expected_warnings_i

    encoding = tiktoken.encoding_for_model(DEFAULT_MODEL)

    def _count_tokens(code: str) -> int:
        return len(encoding.encode(code))

    assert all(c.token_count == _count_tokens(c.code) for c in chunks)

    all_lines = code.split('\n')
    unaccounted_for_lines: Set[int] = set(range(len(all_lines)))

    for chunk in chunks:
        chunk_lines = chunk.code.strip('\n').split('\n')
        for i, chunk_line in itertools.zip_longest(range(chunk.lineno - 1, chunk.end_lineno), chunk_lines):
            assert i is not None
            assert chunk_line is not None
            if i in (chunk.lineno - 1, chunk.end_lineno - 1):
                assert chunk_line in all_lines[i]
            else:
                escaped_chunk_line = re.escape(chunk_line)
                assert re.search(r'^\s*' + escaped_chunk_line + r'$', all_lines[i]), \
                    f"{chunk_line!r} not in {all_lines[i]!r}"

            if i in unaccounted_for_lines:
                unaccounted_for_lines.remove(i)

    for i in unaccounted_for_lines:
        try:
            parsed = ast.parse(all_lines[i])
            child_node = list(ast.iter_child_nodes(parsed))
            assert len(child_node) == 0, f'Line #{i} {all_lines[i]!r} has not been accounted for'
        except SyntaxError:
            raise AssertionError(f'Line #{i} {all_lines[i]!r} has not been accounted for')


@pytest.fixture
def max_tokens_to_send(request: Any) -> int:
    marker = request.node.get_closest_marker("max_tokens_to_send")
    assert_message = 'To use this fixture test function needs to have max_tokens_to_send marker passed one argument' \
                     ' of type int'
    assert marker is not None, assert_message
    assert len(marker.args) == 1, assert_message
    assert isinstance(marker.args[0], int), assert_message

    return marker.args[0]


@pytest.fixture
def system_content() -> str:
    return 'You are a tester, good jorb!!!'


@pytest.fixture
def code() -> str:
    return 'print("Hello World")'


@pytest.fixture
def examples_added_query(examples_file: ExamplesFile, max_tokens_to_send: int, code: str, system_content: str) \
        -> List[Message]:
    mock_open_ai_client = create_autospec(OpenAiClient, instance=True)

    example_embeddings_by_text = {
        e.code: cast(ExampleWithEmbeds, e).embeddings for e in examples_file.examples
    }

    def _get_embedding_for_text(text: str) -> List[float]:
        if text in example_embeddings_by_text:
            return example_embeddings_by_text[text]
        elif text == code:
            return [1.0, 0.0]
        else:
            raise AssertionError(f"Illegal text: {text!r}")

    def _mock_get_embeddings(texts: Iterable[str]) -> Iterable[EmbeddingItem]:
        return_value = [
            EmbeddingItem(t, _get_embedding_for_text(t)) for t in texts
        ]
        Random(777).shuffle(return_value)
        return return_value

    mock_open_ai_client.get_embeddings.side_effect = _mock_get_embeddings

    query_constructor = QueryConstructor(
        open_ai_client=mock_open_ai_client,
        examples=examples_file.examples,
        max_tokens_to_send=max_tokens_to_send,
        system_content=system_content
    )
    query = query_constructor.add_examples_to_query(code)

    return query


@pytest.mark.max_tokens_to_send(90)
def test_add_examples_to_query(examples_added_query: List[Message], system_content: str, examples_file: ExamplesFile,
                               code: str):
    expected_query = [Message(role=Role.system, content=system_content)]
    for expected_example in reversed(examples_file.examples[2:]):
        expected_query.extend([
            Message(role=Role.user, content=expected_example.code),
            Message(role=Role.agent, content=expected_example.response)
        ])
    expected_query.append(
        Message(role=Role.user, content=code)
    )

    assert examples_added_query == expected_query


@pytest.mark.max_tokens_to_send(150)
def test_add_all_examples_to_query(examples_added_query: List[Message], system_content: str,
                                   examples_file: ExamplesFile, code: str):
    expected_query = [Message(role=Role.system, content=system_content)]
    for expected_example in examples_file.examples:
        expected_query.extend([
            Message(role=Role.user, content=expected_example.code),
            Message(role=Role.agent, content=expected_example.response)
        ])
    expected_query.append(
        Message(role=Role.user, content=code)
    )

    assert examples_added_query == expected_query


SNIPPET_FILENAME_PARAMS = [
    '002db0f84128233e76024757725b7af64dfa95af08d2195ffdc74a564b3fce9e_after_merge-normalized.py',
    '0090562c6483883dc4db96053d8674aa47a7d2fb513c7ec573582356f6fc5fee_after_merge-normalized.py'
]


@pytest.fixture(params=SNIPPET_FILENAME_PARAMS)
def snippet_filename(request: Any, base_dir: str) -> str:
    return os.path.join(base_dir, 'tests', 'resources', 'snippets', request.param)


CHUNK_SIZE_PARAMS = [
    DEFAULT_MAX_CHUNK_SIZE,
    DEFAULT_MAX_CHUNK_SIZE // 2,
    DEFAULT_MAX_CHUNK_SIZE // 4,
    DEFAULT_MAX_CHUNK_SIZE * 2
]


@pytest.fixture(params=CHUNK_SIZE_PARAMS)
def max_chunk_size_snippets(request: Any) -> int:
    return request.param


@pytest.fixture(params=CHUNK_SIZE_PARAMS)
def abs_max_chunk_size_snippets(request: Any) -> int:
    return request.param

@pytest.fixture
def chunk_parse_warning() -> str:
    return f"WARNING: {CHUNK_PARSE_MESSAGE}"


def test_chunker_against_snippets(base_dir: str, snippet_filename: str, max_chunk_size_snippets: int,
                                  abs_max_chunk_size_snippets: int, chunk_parse_warning: str):
    """
    Test the chunker against snippets from pytracebugs that gave us trouble
    https://github.com/acheshkov/pytracebugs
    """
    full_path = os.path.join(base_dir, snippet_filename)
    with open(full_path, 'r') as f:
        code = f.read()

    try:
        chunker = CodeChunker(
            code=code,
            file=full_path,
            max_chunk_size=max_chunk_size_snippets,
            abs_max_chunk_size=abs_max_chunk_size_snippets,
            strict_chunk_size=False
        )
    except AssertionError:
        assert abs_max_chunk_size_snippets < max_chunk_size_snippets, \
            'This should have succeeded otherwise'
    else:
        assert abs_max_chunk_size_snippets >= max_chunk_size_snippets, \
            'If creating the chunker succeeded, this should be true'

        chunks = chunker.get_chunks()
        for i, chunk in enumerate(chunks):
            try:
                ast.parse(chunk.code)
            except SyntaxError:
                syntax_error = True
            else:
                syntax_error = False

            if syntax_error:
                assert chunk.warning is not None, f"Error for chunk {chunk}"
            else:
                assert chunk.warning is None or chunk.warning != chunk_parse_warning, \
                    f"There shouldn't be an error for chunk {chunk}"

            if chunk.token_count > abs_max_chunk_size_snippets:
                assert chunk.warning and chunk.warning != chunk_parse_warning, \
                    f"Size warning trumps parsing warning ({chunk})"
            elif syntax_error:
                assert chunk.warning == chunk_parse_warning, \
                    f"If there is a syntax error and no size problems, this should be the warning messsage ({chunk})"
