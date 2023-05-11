"""Main module."""
import ast
import os
import re
from dataclasses import dataclass
from fnmatch import fnmatch
from hashlib import md5
from typing import Iterable, List, Optional, Any, Tuple, NamedTuple, TypeVar, Type, cast
from uuid import uuid4, UUID
import tiktoken

from py_bugs_open_ai.constants import DEFAULT_MODEL
from .models.base import CacheProtocol
from .models.open_ai import Message, Role
from .open_ai_client import OpenAiClient


AstT = TypeVar('AstT', bound=ast.AST)


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
    file: str
    lineno: int
    end_lineno: int
    col_offset: int
    end_col_offset: int
    code: str
    peer_group: UUID
    token_count: int
    error: str | None = None
    warning: str | None = None

    def get_hash(self):
        return md5(self.code.encode()).hexdigest()[:10]

    def set_exception(self, message: str, error: bool) -> 'CodeChunk':
        prefix = 'ERROR' if error else 'WARNING'
        if error:
            self.error = f"{prefix}: {message}"
        else:
            self.warning = f"{prefix}: {message}"
        return self


T = TypeVar('T')


def coalesce(*args: Optional[T]) -> T:
    for arg in args:
        if arg is not None:
            return arg
    raise TypeError('At least one argument needs to not be None')


class CodeChunker(ast.NodeVisitor):
    def __init__(self, code: str, file: str, max_chunk_size: int, model: str = DEFAULT_MODEL,
                 abs_max_chunk_size: int = -1, strict_chunk_size: bool = False):
        self.model = model
        self.max_chunk_size = max_chunk_size
        self.strict_chunk_size = strict_chunk_size
        self.file = file
        if abs_max_chunk_size < 0:
            self.abs_max_chunk_size = self.max_chunk_size
        else:
            self.abs_max_chunk_size = abs_max_chunk_size
        self._chunks_by_peer_group: List[List[CodeChunk]] = []
        self._current_peer_group = uuid4()
        self._code_lines = code.split('\n')
        self._tree = ast.parse(code)

        self.visit(self._tree)

    def get_chunks(self) -> Iterable[CodeChunk]:
        for peer_group in self._chunks_by_peer_group:
            yield from self.chunk_up_peer_group(peer_group)

    def _get_chunk_size_exception_message(self, chunk: CodeChunk) -> str:
        return f"Chunk size {chunk.token_count} bigger than max size {self.abs_max_chunk_size}"

    def chunk_up_peer_group(self, peer_group: List[CodeChunk]) -> Iterable[CodeChunk]:
        if peer_group:
            if len(peer_group) >= 2:
                total_token_count = self.combine_from_to_chunks(peer_group[0], peer_group[-1]).token_count
            else:
                total_token_count = peer_group[0].token_count

            goal_min_size = self.get_goal_min_size(
                total_token_count=total_token_count,
                max_chunk_size=self.max_chunk_size
            )
            last_chunk: CodeChunk | None = None
            for chunk in peer_group:
                if last_chunk is not None:
                    concat_chunk = self.combine_from_to_chunks(last_chunk, chunk)
                else:
                    concat_chunk = chunk

                if concat_chunk.token_count >= goal_min_size:
                    if concat_chunk.token_count <= self.max_chunk_size:
                        yield concat_chunk
                        last_chunk = None
                    elif last_chunk:
                        assert last_chunk.token_count <= self.abs_max_chunk_size
                        yield last_chunk
                        last_chunk = chunk
                    else:
                        assert concat_chunk is chunk, 'These should be the same in this case'
                        if chunk.token_count <= self.abs_max_chunk_size:
                            yield chunk
                        else:
                            chunk = chunk.set_exception(
                                self._get_chunk_size_exception_message(chunk), error=self.strict_chunk_size
                            )
                            yield chunk
                else:
                    last_chunk = concat_chunk

            if last_chunk is not None:
                yield last_chunk

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
                        token_count: Optional[int] = None, peer_group: UUID | None = None) -> CodeChunk:
        lines = self._code_lines[lineno - 1:end_lineno]
        if indent_match := re.search(r'^\s+', lines[0]):
            indent_len = len(indent_match.group(0))
        else:
            indent_len = 0
        lines[-1] = lines[-1][:end_col_offset]
        lines = [
            lines[0][col_offset:],
            *(line[indent_len:] for line in lines[1:])
        ]

        code = '\n'.join(lines) + '\n'
        if token_count is None:
            token_count_ = self.get_token_count(code)
        else:
            token_count_ = token_count

        if peer_group is None:
            peer_group_ = self._current_peer_group
        else:
            peer_group_ = peer_group

        return CodeChunk(
            file=self.file,
            lineno=lineno,
            end_lineno=end_lineno,
            col_offset=col_offset,
            end_col_offset=end_col_offset,
            code=code,
            peer_group=peer_group_,
            token_count=token_count_
        )

    def _get_children(self, node: ast.AST, of_type: Type[ast.AST] = ast.AST) -> Iterable[ast.AST]:
        for field, value in ast.iter_fields(node):
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, of_type):
                        yield item
            elif isinstance(value, of_type):
                yield value

    def _get_stmt_header(self, node: ast.stmt) -> Tuple[CodeChunk | None, List[ast.stmt]]:
        lineno = node.lineno
        col_offset = node.col_offset
        sub_stmts = cast(List[ast.stmt], list(self._get_children(node, of_type=ast.stmt)))
        if len(sub_stmts) > 0:
            end_lineno, end_col_offset = min((n.lineno, n.col_offset) for n in sub_stmts)
            assert (lineno, col_offset) <= (end_lineno, end_col_offset)

            return_chunk = self.make_code_chunk(
                lineno=lineno,
                col_offset=col_offset,
                end_lineno=end_lineno,
                end_col_offset=end_col_offset,
                peer_group=uuid4()
            )
        else:
            return_chunk = None

        return return_chunk, sub_stmts

    def generic_visit(self, node) -> Any:
        """
        Note: this should probably not use NodeVisitor anymore, and then also pass values like peer_group
        in a more functional manner.  TODO
        """
        chunk = self.chunk_from_node(node)
        new_peer_group: UUID | None
        children_to_visit: List[ast.AST]
        if chunk is not None and chunk.token_count <= self.max_chunk_size:
            if self._chunks_by_peer_group \
                    and self._chunks_by_peer_group[-1][0].peer_group == chunk.peer_group:
                peer_group = self._chunks_by_peer_group[-1]
            else:
                peer_group = []
                self._chunks_by_peer_group.append(peer_group)
            peer_group.append(chunk)
            children_to_visit = []
            new_peer_group = uuid4()
        elif chunk and isinstance(node, ast.stmt):
            # add chunk to child peer group
            header_chunk, children_to_visit_ = self._get_stmt_header(node)
            children_to_visit = cast(List[ast.AST], children_to_visit_)
            if header_chunk is not None:
                self._chunks_by_peer_group.append([header_chunk])
                new_peer_group = header_chunk.peer_group
            else:
                # might be too big, will determine to warn or error in .chunk_up_peer_group()
                chunk.peer_group = uuid4()
                self._chunks_by_peer_group.append([chunk])
                children_to_visit = []
                new_peer_group = None
            # chunk = self.collapse_chunk(chunk)
        elif chunk:
            raise AssertionError('This shouldn\'t happen, if chunk is not None then it is a stmt')
        else:
            children_to_visit = list(self._get_children(node))
            new_peer_group = uuid4()

        if len(children_to_visit) > 0:
            assert new_peer_group is not None, 'If we get here, new_peer_group should have been set'
            old_peer_group = self._current_peer_group
            try:
                self._current_peer_group = new_peer_group
                super().generic_visit(node)
            finally:
                self._current_peer_group = old_peer_group

    def combine_from_to_chunks(self, chunk_a: CodeChunk, chunk_b: CodeChunk) -> CodeChunk:
        """
        Assumes the code chunks are consecutive.  If they aren't, it will capture the code in-between
        """
        assert chunk_a.end_lineno <= chunk_b.lineno, 'chunk_a should be before chunk_b in the code'
        # remake the chunk from the linenos and offsets so we get the spaces between
        return self.make_code_chunk(
            lineno=chunk_a.lineno,
            end_lineno=chunk_b.end_lineno,
            col_offset=chunk_a.col_offset,
            end_col_offset=chunk_b.end_col_offset
        )


class FindBugsReturn(NamedTuple):
    is_bug: bool
    description: str


class BugFinder:
    FIND_BUGS_SYSTEM_CONTENT = \
        'You are a python bug finder.  Given a snippet of python code, you respond "OK" if you detect no bugs in it' \
        ' and"ERROR: " followed by the error description if you detect an error in it.  Don\'t report import errors' \
        ' packages.'

    def __init__(self, model: str, api_key: str, cache: CacheProtocol[str, str] | None = None,
                 is_bug_re: re.Pattern | None = None, system_content: str = FIND_BUGS_SYSTEM_CONTENT):
        cache_ = cache if cache is not None else {}
        self.open_ai_client = OpenAiClient(api_key, model=model, cache=cache_)
        self.is_bug_re = is_bug_re if is_bug_re is not None else re.compile(r'^ERROR\b')
        self.system_content = system_content

    def get_query_messages(self, code: str) -> List[Message]:
        return [
            Message(Role.system, self.system_content),
            Message(Role.user, code),
        ]

    def find_bugs(self, code: str, refresh_cache: bool = False) -> FindBugsReturn:
        query_messages = self.get_query_messages(code)
        description = self.open_ai_client.query_messages(query_messages, refresh_cache=refresh_cache)
        is_bug = bool(self.is_bug_re.search(description))

        return FindBugsReturn(is_bug, description)
