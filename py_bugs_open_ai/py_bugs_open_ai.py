"""Main module."""
import ast
import itertools
import re
from dataclasses import dataclass, field
from hashlib import md5
from math import ceil
from typing import Iterable, List, Optional, Any, Tuple, NamedTuple, TypeVar, Type, cast, Callable
from uuid import uuid4, UUID
import tiktoken
from scipy import spatial  # type: ignore

from .constants import DEFAULT_MODEL, DEFAULT_IS_BUG_RE, FIND_BUGS_SYSTEM_CONTENT
from .models.base import CacheProtocol
from .models.examples import Example
from .models.open_ai import Message, Role
from .open_ai_client import OpenAiClient

CHUNK_PARSE_MESSAGE = 'Unable to parse chunk'

AstT = TypeVar('AstT', bound=ast.AST)


def _cosine_wrapper(u: List[float], v: List[float]) -> float:
    # wrapper to correctly type spatial.distance.cosine()
    return spatial.distance.cosine(u, v)


class CodeChunkException(Exception):
    def __init__(self, message: str, is_error: bool):
        super().__init__()
        self.message = message
        self.is_error = is_error


class ChunkSizeException(CodeChunkException):
    def __init__(self, token_count: int, max_size: int, is_error: bool):
        error_or_warning = 'ERROR' if is_error else 'WARNING'
        message = f"{error_or_warning}: Chunk size {token_count} bigger than max size {max_size}"
        super().__init__(message=message, is_error=is_error)


class ChunkErrorFoundException(CodeChunkException):
    def __init__(self, message: str):
        super().__init__(message=message, is_error=True)


class ChunkParseException(CodeChunkException):
    def __init__(self):
        super().__init__(message='WARNING: Unable to parse chunk', is_error=False)


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
    exceptions: List[CodeChunkException] = field(default_factory=lambda: [])

    def get_hash(self):
        return md5(self.code.encode()).hexdigest()[:10]

    def check_is_valid(self) -> bool:
        """
        True if the code represented by this chunk is valid python code
        """
        try:
            ast.parse(self.code)
        except SyntaxError:
            return False
        else:
            return True

    def has_exception(self, is_error: Optional[bool] = None,
                      of_type: Optional[Type[CodeChunkException]] = None) -> bool:
        """
        Says whether this chunk has any warnings
        If is_error set, returns True only if any of the exeptions have .is_error set to the same value
        if of_type is set, only returns true only if any of the exceptions are of type of_type
        """
        def _is_error_check(exception_: CodeChunkException) -> bool:
            if is_error is None:
                return True
            elif exception_.is_error is is_error:
                return True
            else:
                return False

        def _of_type_check(exception_: CodeChunkException) -> bool:
            if of_type is None:
                return True
            elif isinstance(exception_, of_type):
                return True
            else:
                return False

        for exception in self.exceptions:
            if _is_error_check(exception) and _of_type_check(exception):
                return True
        return False


T = TypeVar('T')


def coalesce(*args: Optional[T]) -> T:
    for arg in args:
        if arg is not None:
            return arg
    raise TypeError('At least one argument needs to not be None')


def assert_strict(is_true: bool, message: str = 'ERROR') -> None:
    """
    Assert that fails regardless of whether production key is set
    """
    if not is_true:
        raise AssertionError(message)


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
        assert_strict(self.abs_max_chunk_size >= self.max_chunk_size)

        self._current_peer_group = uuid4()  # used in _populate_chunks_by_peer_group()
        self._code_lines = code.split('\n')
        self._tree = ast.parse(code)
        # see .generic_visit() for how this is populated/what this is
        self._chunks_by_peer_group: List[List[CodeChunk]] = []

        self._populate_chunks_by_peer_group()

    def _populate_chunks_by_peer_group(self):
        # just calls .visit(), but trying to make clear calling .visit() will populate ._chunks_by_peer_group
        # .visit() ends up calling .generic_visit(), where the magic happens
        self.visit(self._tree)

    def get_chunks(self) -> Iterable[CodeChunk]:
        """
        The algorythm for chunking the code is roughly as follows:
            - Divide the code into "peer_groups", which will contain chunks that it thinks makes sense to keep together
                - Already done in __init__()
            - Tries to merge those peer groups so they're smaller than .max_chunk_size but biggest possible
                - Sometimes we are left with chunks bigger than .max_chunk_size.  Then, if the chunk is also
                  bigger than .abs_max_chunk_size, we add a message to the .warning or .error of the chunk, depending
                  on .strict_chunk_size
                - We also try to keep these chunks as legitimate, compilable python code.  If we can't, we also
                  add a ChunkParseException() to exceptions
        """
        for peer_group in self._chunks_by_peer_group:
            # for each peer group, try to merge the chunks into the largest possible which is less than the
            # .max_chunk_size
            yield from self.merge_peer_group_chunks(peer_group)

    def merge_peer_group_chunks(self, peer_group: List[CodeChunk]) -> Iterable[CodeChunk]:
        """
        Goes through and merges the chunks in the peer groups to be biggest possible but less than
        max_chunk_size.

        Algorythm is roughly:
        goal_min_size = see .get_goal_min_size()
        last_chunk = empty
        for chunk in peer_group:
            concat_chunk = last_chunk + chunk
            if concat_chunk > goal_min_size:
                if concat_chunk < max_chunk_size and concat_chunk.is_valid():
                    yield concat_chunk
                    clear last_chunk
                elif last_chunk and last_chunk.is_valid():
                    yield last_chunk
                    last_chunk = chunk
                elif concat_chunk > abs_max_chunk_size:
                    yield concat_chunk with error or warning
                    clear last_chunk
                else:
                    last_chunk = concat_chunk
            else:
                last_chunk = concat_chunk

        if last_chunk:
            yield last_chunk with to-big or parse exceptions
        """
        if peer_group:
            if len(peer_group) >= 2:
                total_token_count = self.combine_from_to_chunks(peer_group[0], peer_group[-1]).token_count
            else:
                total_token_count = peer_group[0].token_count

            goal_min_size = self.get_goal_min_size(
                total_token_count=total_token_count,
                max_chunk_size=self.max_chunk_size
            )
            last_chunk: Optional[CodeChunk] = None
            for chunk in peer_group:
                if last_chunk is not None:
                    concat_chunk = self.combine_from_to_chunks(last_chunk, chunk)
                else:
                    concat_chunk = chunk

                if concat_chunk.token_count >= goal_min_size:
                    if concat_chunk.token_count <= self.max_chunk_size and concat_chunk.check_is_valid():
                        yield concat_chunk
                        last_chunk = None
                    elif last_chunk is not None and last_chunk.check_is_valid():
                        assert_strict(last_chunk.token_count <= self.abs_max_chunk_size)
                        yield last_chunk
                        last_chunk = chunk
                    elif concat_chunk.token_count > self.abs_max_chunk_size:
                        concat_chunk.exceptions.append(ChunkSizeException(
                            token_count=concat_chunk.token_count,
                            max_size=self.max_chunk_size,
                            is_error=self.strict_chunk_size
                        ))
                        yield concat_chunk
                        last_chunk = None
                    else:
                        last_chunk = concat_chunk
                else:
                    last_chunk = concat_chunk

            if last_chunk is not None:
                if not last_chunk.check_is_valid():
                    last_chunk.exceptions.append(ChunkParseException())
                if last_chunk.token_count > self.abs_max_chunk_size:
                    last_chunk.exceptions.append(ChunkSizeException(
                        token_count=last_chunk.token_count,
                        max_size=self.max_chunk_size,
                        is_error=self.strict_chunk_size
                    ))

                yield last_chunk

    @staticmethod
    def get_goal_min_size(total_token_count: int, max_chunk_size: int) -> int:
        """
        Given the total count and the max size, finds the way to break up the chunk which will have roughly
        equal sized chunks
        """
        goal_num_chunks = ceil(total_token_count / max_chunk_size)
        goal_min_size = round(total_token_count / goal_num_chunks)

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
                        token_count: Optional[int] = None, peer_group: Optional[UUID] = None) -> CodeChunk:
        """
        Makes a code chunk from the metadata passed
        """
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
        for _, value in ast.iter_fields(node):
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, of_type):
                        yield item
            elif isinstance(value, of_type):
                yield value

    def _get_stmt_header(self, node: ast.stmt) -> Tuple[Optional[CodeChunk], List[ast.stmt]]:
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
        Populates ._chunk_by_peer_group.  This is a list of lists, with each list representing a "peer group", meaning
        peers in the same node.  If one of the peers is > max_chunk_size, it descends the tree to the next
        peer group

        Note: If I were to write this from scratch, I would probably not use ast.NodeVisitor and do this more,
        functionally, since it doesn't use much of ast.NodeVisitor's functionality and has to store information
        by changing the state of the instance, which I hate
        """
        chunk = self.chunk_from_node(node)
        new_peer_group: Optional[UUID]
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
                # might be too big, will determine to warn or error in .merge_peer_group_chunks()
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
    def __init__(self, open_ai_client: OpenAiClient, is_bug_re: Optional[re.Pattern] = None,
                 system_content: str = FIND_BUGS_SYSTEM_CONTENT):
        self.open_ai_client = open_ai_client
        self.is_bug_re = is_bug_re if is_bug_re is not None else re.compile(DEFAULT_IS_BUG_RE)
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


class QueryConstructor:
    def __init__(self, open_ai_client: OpenAiClient, examples: List[Example], max_tokens_to_send: int,
                 system_content: str = FIND_BUGS_SYSTEM_CONTENT, model: str = DEFAULT_MODEL):
        self.open_ai_client = open_ai_client
        self.max_tokens_to_send = max_tokens_to_send
        self.system_content = system_content
        self.model = model  # for getting token count
        self._token_count_cache: CacheProtocol[str, int] = {}
        self.examples = examples

    def get_token_count(self, code: str, refresh_cache: bool = False) -> int:
        """Return the number of tokens in a string."""
        if refresh_cache or code not in self._token_count_cache:
            encoding = tiktoken.encoding_for_model(self.model)
            self._token_count_cache[code] = len(encoding.encode(code))
        return self._token_count_cache[code]

    def _get_token_count_sum(self, messages: List[Message]) -> int:
        return sum(self.get_token_count(m.content) for m in messages)

    def _get_starting_messages(self, query: str) -> List[Message]:
        return [
            Message(role=Role.system, content=self.system_content),
            Message(role=Role.user, content=query),
        ]

    def add_examples_to_query(self, query: str) -> List[Message]:
        filter_examples = self.will_filter_examples(query)

        if filter_examples:
            return self._add_examples_filtered(query)
        else:
            return self._add_examples_all(query)

    def will_filter_examples(self, query: str) -> bool:
        starting_messages = self._get_starting_messages(query)
        token_count = self._get_token_count_sum(starting_messages)
        for example in self.examples:
            token_count += self.get_token_count(example.code)
            token_count += self.get_token_count(example.response)
            if token_count > self.max_tokens_to_send:
                filter_examples = True
                break
        else:
            filter_examples = False
        return filter_examples

    def _add_examples_all(self, query: str) -> List[Message]:
        starting_messages = self._get_starting_messages(query)
        return_messages = starting_messages[:-1]
        for example in self.examples:
            return_messages.append(Message(
                role=Role.user,
                content=example.code,
            ))
            return_messages.append(Message(
                role=Role.agent,
                content=example.response
            ))
        return_messages.append(starting_messages[-1])
        return return_messages

    @staticmethod
    def _sorted(to_sort: Iterable[T], key: Callable[[T], Any]) -> Iterable[T]:
        to_sort_keyed = map(lambda x: (key(x), x), to_sort)
        sorted_keyed = sorted(to_sort_keyed, key=lambda x: x[0])
        yield from map(lambda x: x[1], sorted_keyed)

    def _add_examples_filtered(self, query: str) -> List[Message]:
        starting_messages = self._get_starting_messages(query)
        texts_iter = itertools.chain((e.code for e in self.examples), [query])
        embeddings = self.open_ai_client.get_embeddings(texts=texts_iter)  # this should be cached
        embeddings_by_text = {text: embeddings for text, embeddings in embeddings}

        query_embeddings = embeddings_by_text[query]

        def _rank(example_: Example):
            return _cosine_wrapper(query_embeddings, embeddings_by_text[example_.code])
        sorted_examples = self._sorted(self.examples, key=_rank)

        return_messages = starting_messages
        token_count = self._get_token_count_sum(starting_messages)
        for example in sorted_examples:
            token_count += self.get_token_count(example.code) + self.get_token_count(example.response)
            if token_count > self.max_tokens_to_send:
                return return_messages  # return examples without latest
            return_messages = [
                *return_messages[:-1],
                Message(role=Role.user, content=example.code),
                Message(role=Role.agent, content=example.response),
                return_messages[-1]
            ]
        return return_messages  # we shouldn't get here, but :shrug:
