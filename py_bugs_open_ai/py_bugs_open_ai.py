"""Main module."""
import ast
import json
import os
import re
import time
from dataclasses import dataclass
from fnmatch import fnmatch
from typing import Iterable, List, Optional, Any, Tuple, NamedTuple, Dict, TypeVar
from uuid import uuid4, UUID
import tiktoken

from .models.base import CacheProtocol
from .models.open_ai import Message, Role
from .open_ai_client import OpenAiClient



def get_files(path: str = '.', include: str = '*.py', exclude: Optional[List[str]] = None) -> Iterable[str]:
    if exclude is not None:
        exclude_ = exclude
    else:
        exclude_ = []

    def _is_excluded(name: str, is_dir: bool) -> bool:
        for exclude_pattern in exclude_:
            if is_dir and exclude_pattern.endswith('/'):
                exclude_pattern = exclude_pattern[:-1]
            if fnmatch(name, exclude_pattern):
                return True
        return False

    if os.path.isfile(path):
        if fnmatch(path, include) and not _is_excluded(path, is_dir=False):
            yield path
    else:
        dir = path
        dirs_to_search: List[str] = []
        while True:
            for dir_item in os.listdir(dir):
                full_item = os.path.join(dir, dir_item)
                if os.path.isdir(full_item) and not _is_excluded(dir_item, is_dir=True):
                   dirs_to_search.append(full_item)
                elif fnmatch(dir_item, include) and not _is_excluded(dir_item, is_dir=False):
                    yield full_item
            if dirs_to_search:
                dir = dirs_to_search.pop()
            else:
                break


@dataclass
class CodeChunk:
    lineno: int
    end_lineno: int
    col_offset: int
    end_col_offset: int
    code: str
    peer_group: UUID
    token_count: int


T = TypeVar('T')


def coalesce(*args: Optional[T]) -> T:
    for arg in args:
        if arg is not None:
            return arg
    raise TypeError('At least one argument needs to not be None')


class CodeChunker(ast.NodeVisitor):
    NODE_TYPES_TO_CHUNK = [
        ast.FunctionDef, ast.AsyncFunctionDef, ast.Import, ast.ImportFrom,
        ast.With, ast.AsyncWith, ast.For, ast.With, ast.ClassDef, ast.Assign, ast.Assert
    ]

    def __init__(self, code: str, max_chunk_size: int, model: str):
        self.model = model
        self.max_chunk_size = max_chunk_size
        self._chunks_by_peer_group: List[List[CodeChunk]] = []
        self._current_peer_group = uuid4()
        self._code_lines = code.split('\n')
        self._tree = ast.parse(code)

        self.visit(self._tree)

    def get_chunks(self) -> Iterable[CodeChunk]:
        for peer_group in self._chunks_by_peer_group:
            yield from self.chunk_up_peer_group(peer_group)

    def chunk_up_peer_group(self, peer_group: List[CodeChunk]) -> Iterable[CodeChunk]:
        if peer_group:
            total_token_count = sum(c.token_count for c in peer_group)
            goal_min_size = self.get_goal_min_size(
                total_token_count=total_token_count,
                max_chunk_size=self.max_chunk_size
            )
            current_chunk: Optional[CodeChunk] = None
            for chunk in peer_group:
                if current_chunk is None:
                    current_chunk = chunk
                elif current_chunk is not None and current_chunk.token_count + chunk.token_count < self.max_chunk_size:
                    current_chunk = self.concat_consec_code_chunks(current_chunk, chunk)
                    if current_chunk.token_count >= goal_min_size:
                        yield current_chunk
                        current_chunk = None
                elif current_chunk is not None:
                    yield current_chunk
                    current_chunk = chunk
            if current_chunk is not None:
                yield current_chunk

    @staticmethod
    def get_goal_min_size(total_token_count: int, max_chunk_size: int) -> int:
        goal_min_size = total_token_count  # will try to get each chunk up to this size
        goal_num_chunks = 1
        while goal_min_size > max_chunk_size:
            goal_num_chunks += 1
            goal_min_size = total_token_count // goal_num_chunks
        return goal_min_size

    def get_token_count(self, code: str) -> int:
        """Return the number of tokens in a string."""
        encoding = tiktoken.encoding_for_model(self.model)
        return len(encoding.encode(code))

    def chunk_from_node(self, node: ast.AST) -> Optional[CodeChunk]:
        if isinstance(node, ast.stmt):
                # and any(isinstance(node, type_) for type_ in self.NODE_TYPES_TO_CHUNK):
            return self.make_code_chunk(
                lineno=coalesce(node.lineno, 0),
                end_lineno=coalesce(node.end_lineno, 0),
                col_offset=coalesce(node.col_offset, 0),
                end_col_offset=coalesce(node.end_col_offset, 0)
            )
        else:
            return None

    def make_code_chunk(self, lineno: int, end_lineno: int, col_offset: int, end_col_offset: int,
                        token_count: Optional[int] = None) -> CodeChunk:
        lines = self._code_lines[lineno - 1:end_lineno]
        if indent_match := re.search(r'^\s+', lines[0]):
            indent_len = len(indent_match.group(0))
        else:
            indent_len = 0
        lines[-1] = lines[-1][:end_col_offset]
        lines = [
            lines[0][col_offset:],
            *(l[indent_len:] for l in lines[1:])
        ]

        code = '\n'.join(lines) + '\n'
        if token_count is None:
            token_count_ = self.get_token_count(code)
        else:
            token_count_ = token_count

        return CodeChunk(
            lineno=lineno,
            end_lineno=end_lineno,
            col_offset=col_offset,
            end_col_offset=end_col_offset,
            code=code,
            peer_group=self._current_peer_group,
            token_count=token_count_
        )

    def generic_visit(self, node: ast.AST) -> Any:
        chunk = self.chunk_from_node(node)
        if isinstance(node, ast.ClassDef):
            pass
        if chunk is not None and chunk.token_count <= self.max_chunk_size:
            if self._chunks_by_peer_group \
                    and self._chunks_by_peer_group[-1][0].peer_group == chunk.peer_group:
                peer_group = self._chunks_by_peer_group[-1]
            else:
                peer_group = []
                self._chunks_by_peer_group.append(peer_group)
            if chunk.token_count > self.max_chunk_size:
                visit_children = True
                # add chunk to child peer group
                chunk.peer_group = uuid4()
                self._chunks_by_peer_group.append([chunk])
                # chunk = self.collapse_chunk(chunk)
            else:
                visit_children = False
                peer_group.append(chunk)
            # peer_group.append(chunk)
        else:
            visit_children = True

        if visit_children:
            old_peer_group = self._current_peer_group
            try:
                self._current_peer_group = chunk.peer_group if chunk is not None else uuid4()
                super().generic_visit(node)
            finally:
                self._current_peer_group = old_peer_group

    # def collapse_chunk(self, chunk: CodeChunk) -> CodeChunk:
    #     """ TODO """
    #     return CodeChunk(
    #         code=f'**** TO COLAPSE {chunk.token_count} TOKENS ****',
    #         lineno=0,
    #         end_lineno=0,
    #         col_offset=0,
    #         end_col_offset=0
    #
    #     )

    def concat_consec_code_chunks(self, chunk_a: CodeChunk, chunk_b: CodeChunk) -> CodeChunk:
        """
        Assumes the code chunks are consecutive.  If they aren't, it will capture the code in-between
        """
        assert chunk_a.end_lineno <= chunk_b.lineno, 'chunk_a should be before chunk_b in the code'
        # remake the chunk from the linenos and offsets so we get the spaces between
        return self.make_code_chunk(
            lineno=chunk_a.lineno,
            end_lineno=chunk_b.end_lineno,
            col_offset=chunk_a.col_offset,
            end_col_offset=chunk_b.end_col_offset,
            token_count=chunk_a.token_count + chunk_b.token_count
        )


open_ai_client: Optional[OpenAiClient] = None

def get_find_bug_query(code: str) -> str:
    return_text = '##### Are there bugs in the following Python code?  Respond in json with the following format:' \
                  ' {"has_bugs": true or false, "bugs_description": string of bug descriptions}\n\n'
    return_text += code
    return_text += '\n\n### __RESPONSE__:'

    return return_text

def get_api_key():
    # at some point something secure here
    with open('/Users/valmikirao/.ssh/openapi-key.txt', 'r') as f:
        openai_key = f.read()
    return openai_key.strip()

class FindBugsReturn(NamedTuple):
    is_bug: bool
    description: str


class BugFinder:
    FIND_BUGS_SYSTEM_CONTENT = \
        'You are a python bug finder.  Given a snippet of python code, you respond "OK" if you detect no bugs in it' \
        ' and"ERROR: " followed by the error description if you detect an error in it.  Don\'t report import errors' \
        ' packages.'

    def __init__(self, model: str, api_key: str, cache: CacheProtocol[str, str] | None = None, is_bug_re: re.Pattern | None = None,
                 system_content: str = FIND_BUGS_SYSTEM_CONTENT):
        self.open_ai_client = OpenAiClient(api_key, model=model, cache=cache)
        self.is_bug_re = is_bug_re if is_bug_re is not None else re.compile(r'^ERROR\b')
        self.system_content = system_content

    def get_query_messages(self, code: str) -> List[Message]:
        return [
            Message(Role.system, self.system_content),
            Message(Role.user, code),
        ]

    def find_bugs(self, code: str, refresh_cache: bool = False) -> FindBugsReturn:
        query_messages = self.get_query_messages(code)
        description = self.open_ai_client.query_messages(query_messages)
        is_bug = bool(self.is_bug_re.search(description))

        return FindBugsReturn(is_bug, description)


if __name__ == '__main__':
    with open('/Users/valmikirao/Dropbox/git/aws_cloudwatch_insights/aws_cloudwatch_insights/aws_cloudwatch_insights.py', 'r') as f:
        code_chunker = CodeChunker(code=f.read(), max_chunk_size=500, model='gpt-3.5-turbo')
    for chunk in code_chunker.get_chunks():
        print('-' * 80)
        print(chunk.code)
