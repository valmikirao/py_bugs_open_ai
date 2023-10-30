from copy import deepcopy
from dataclasses import dataclass, field
from typing import Dict, List, Iterable, Tuple, Mapping, Final, Sequence, Set, Optional
from unittest.mock import create_autospec
from uuid import uuid4, UUID

import pytest

from py_bugs_open_ai.cli import group_chunks_from_ai, GroupedChunks, should_skip_because_not_in_diff
from py_bugs_open_ai.py_bugs_open_ai import CodeChunk, BugFinder, ChunkErrorFoundException, ChunkSizeException, \
    ChunkParseException
from py_bugs_open_ai.utils import assert_strict


def get_find_bugs_side_effect(bugs_by_i: Dict[int, str]) -> Iterable[Tuple[bool, Optional[str]]]:
    i = 0
    while True:
        if i in bugs_by_i:
            yield True, bugs_by_i[i]
        else:
            yield False, None
        i += 1


class CodeChunkTestList(List[CodeChunk]):
    """
    A list of codechecks such that
    list_0 + list_1 = list_2 a list where
        list_2 is all new instances of the objects in list_0 and list_1
        the linenos in list_2 are contiguous, having updated the copies of the first item from list_1
            to start after the last item in list_2.
    list_0 * int_ = list_2 that's also contiguous with new instances
    - See test for examples
    """

    # Note: for some reason, mypy typechecking doesn't work with these operator overloaders if I
    # put types in the signature

    def __add__(self, other):
        added_list = deepcopy(self)
        assert_strict(isinstance(other, CodeChunkTestList))
        assert isinstance(other, CodeChunkTestList)
        if len(other):
            self_last_lineno = added_list[-1].end_lineno if len(added_list) > 0 else 0
            other_first_lineno = other[0].lineno
            if other_first_lineno < self_last_lineno + 1:
                other_shift = self_last_lineno + 1 - other_first_lineno
            else:
                other_shift = 0
            for chunk in other:
                new_chunk = deepcopy(chunk)
                new_chunk.lineno += other_shift
                new_chunk.end_lineno += other_shift
                added_list.append(new_chunk)

        return added_list

    def __mul__(self, other):
        assert_strict(isinstance(other, int))
        assert isinstance(other, int)
        mul_list = self.__class__([])
        for _ in range(other):
            mul_list = mul_list + self

        return mul_list


ListLineNos = List[Tuple[int, int]]

PEER_GROUP = UUID('b97c2263-afe3-4e25-a6e4-2d50002c31dd')


def code_chunk_test_list_from_linenos(linenos: ListLineNos) -> CodeChunkTestList:
    preliminary_list: List[CodeChunk] = []
    for lineno, end_lineno in linenos:
        preliminary_list.append(CodeChunk(
            file='test-chunk.py',
            lineno=lineno,
            end_lineno=end_lineno,
            col_offset=0,
            end_col_offset=0,
            code='print("Hello World")',
            peer_group=PEER_GROUP,
            token_count=10,
            exceptions=[]
        ))

    return CodeChunkTestList(preliminary_list)


@pytest.mark.parametrize('list_0_linenos,list_1_linenos,expected_linenos', [
    ([(1, 5), (6, 10)], [(1, 5), (6, 10)], [(1, 5), (6, 10), (11, 15), (16, 20)]),
    ([(21, 25), (26, 30)], [(24, 26), (27, 100)], [(21, 25), (26, 30), (31, 33), (34, 107)]),
    ([(1, 5), (6, 10)], [], [(1, 5), (6, 10)]),
    ([], [(1, 5), (6, 10)], [(1, 5), (6, 10)]),
    ([], [], []),
    ([(1, 5), (6, 10)], [(21, 25), (26, 30)], [(1, 5), (6, 10), (21, 25), (26, 30)])
])
def test_code_chunk_test_list_add(list_0_linenos: ListLineNos, list_1_linenos: ListLineNos,
                                  expected_linenos: ListLineNos):

    list_0 = code_chunk_test_list_from_linenos(list_0_linenos)
    list_1 = code_chunk_test_list_from_linenos(list_1_linenos)
    expected = code_chunk_test_list_from_linenos(expected_linenos)

    actual = list_0 + list_1

    assert actual == expected

    # assert something that should always be true for this addition
    assert len(actual) == len(list_0) + len(list_1)

    for i in range(len(list_0) + len(list_1)):
        if i < len(list_0):
            assert actual[i] == list_0[i]
            assert actual[i] is not list_0[0]
        else:
            assert actual[i] is not list_1[i - len(list_0)]

    if len(list_0) > 0 and len(list_1) > 0:
        list_0_end_lineno = list_0[-1].end_lineno
        list_1_start_lineno = list_1[0].lineno
        new_list_1_start_lineno = actual[len(list_0)].lineno

        assert new_list_1_start_lineno > list_0_end_lineno
        if list_1_start_lineno <= list_0_end_lineno:
            assert new_list_1_start_lineno == list_0_end_lineno + 1


@pytest.mark.parametrize('list_0_linenos,n,expected_linenos', [
    ([(1, 5), (6, 10)], 2, [(1, 5), (6, 10), (11, 15), (16, 20)]),
    ([], 5, []),
    ([(1, 5), (6, 10)], 0, []),
    ([(1, 5), (6, 10)], 1, [(1, 5), (6, 10)]),
    ([(10, 20), (21, 25)], 3, [(10, 20), (21, 25), (26, 36), (37, 41), (42, 52), (53, 57)]),
    ([(1, 5), (6, 10)], 100, [(i * 5 + 1, (i + 1) * 5) for i in range(200)])
])
def test_code_chunk_test_list_mul(list_0_linenos: ListLineNos, n: int, expected_linenos: ListLineNos):
    list_0 = code_chunk_test_list_from_linenos(list_0_linenos)
    expected = code_chunk_test_list_from_linenos(expected_linenos)

    actual = list_0 * n

    assert actual == expected


CHUNK_0 = CodeChunk(
    file='test-0.py',
    lineno=1,
    end_lineno=5,
    col_offset=0,
    end_col_offset=0,
    code='print("Hello World")',
    peer_group=uuid4(),
    token_count=10,
    exceptions=[]
)
CHUNK_1 = CHUNK_0.replace(file='test-1.py', code='raise Exception("Hello Universe")')
CHUNK_2 = CHUNK_0.replace(file=CHUNK_1.file, code='assert True, "Hello Multiverse"')
CHUNK_SIZE_WARN = CHUNK_0.replace(exceptions=[ChunkSizeException(token_count=10, max_size=5, is_error=False)])
CHUNK_SIZE_WARN_1 = CHUNK_SIZE_WARN.replace(file=CHUNK_1.file)
CHUNK_SIZE_ERROR = CHUNK_0.replace(exceptions=[ChunkSizeException(token_count=10, max_size=5, is_error=True)])
CHUNK_PARSE_ERROR = CHUNK_0.replace(code='WTF?', exceptions=[ChunkParseException()])
CHUNK_0_LIST = CodeChunkTestList([CHUNK_0])
CHUNK_1_LIST = CodeChunkTestList([CHUNK_1])
CHUNK_2_LIST = CodeChunkTestList([CHUNK_2])
CHUNK_SIZE_WARN_LIST = CodeChunkTestList([CHUNK_SIZE_WARN])
CHUNK_SIZE_ERROR_LIST = CodeChunkTestList([CHUNK_SIZE_ERROR])
CHUNK_PARSE_ERROR_LIST = CodeChunkTestList([CHUNK_PARSE_ERROR])
CHUNK_SIZE_WARN_LIST_1 = CodeChunkTestList([CHUNK_SIZE_WARN_1])

ERROR_0: Final[str] = 'ERROR: Zero'
ERROR_1: Final[str] = 'ERROR: One'

TEST_0_CHUNKS = {'test-0.py': CHUNK_0_LIST * 10}
TEST_1_CHUNKS = {'test-1.py': CHUNK_1_LIST * 10}


@dataclass
class TestGroupChunksAiParametrize:
    chunks_by_file: Mapping[str, List[CodeChunk]] = field(default_factory=lambda: deepcopy(TEST_0_CHUNKS))
    find_bugs_side_effect: Mapping[int, str] = field(default_factory=lambda: {})
    expected: GroupedChunks = field(default_factory=lambda: GroupedChunks())
    die_after: int = 3
    skip_chunks: Set[str] = field(default_factory=lambda: set())
    line_diffs_by_file: Optional[Dict[str, Set[int]]] = None

    @classmethod
    def _get_fields(cls) -> Sequence[str]:
        return list(cls.__dataclass_fields__.keys())

    # not using NamedTuple because of reference weirdness, but then have to implement some function myself
    def __iter__(self):
        return (getattr(self, f) for f in self._get_fields())

    def __len__(self) -> int:
        return len(self._get_fields())

    @classmethod
    def parametrize(cls):
        return pytest.mark.parametrize(cls._get_fields(), [
            cls(),
            cls(find_bugs_side_effect={5: ERROR_0}, expected=GroupedChunks(error_chunks=[
                CHUNK_0.replace(lineno=26, end_lineno=30, exceptions=[ChunkErrorFoundException(ERROR_0)])
            ])),
            cls(find_bugs_side_effect={3: ERROR_0, 8: ERROR_1}, expected=GroupedChunks(error_chunks=[
                CHUNK_0.replace(lineno=16, end_lineno=20, exceptions=[ChunkErrorFoundException(ERROR_0)]),
                CHUNK_0.replace(lineno=41, end_lineno=45, exceptions=[ChunkErrorFoundException(ERROR_1)]),
            ])),
            cls(
                chunks_by_file={**deepcopy(TEST_0_CHUNKS), **deepcopy(TEST_1_CHUNKS)},
                find_bugs_side_effect={5: ERROR_0, 15: ERROR_1},
                expected=GroupedChunks(error_chunks=[
                    CHUNK_0.replace(lineno=26, end_lineno=30, exceptions=[ChunkErrorFoundException(ERROR_0)]),
                    CHUNK_1.replace(lineno=26, end_lineno=30, exceptions=[ChunkErrorFoundException(ERROR_1)])
                ])
            ),
            cls(
                find_bugs_side_effect={3: ERROR_0, 4: ERROR_1, 5: ERROR_0, 6: ERROR_1},
                expected=GroupedChunks(error_chunks=[
                    CHUNK_0.replace(lineno=16, end_lineno=20, exceptions=[ChunkErrorFoundException(ERROR_0)]),
                    CHUNK_0.replace(lineno=21, end_lineno=25, exceptions=[ChunkErrorFoundException(ERROR_1)]),
                    CHUNK_0.replace(lineno=26, end_lineno=30, exceptions=[ChunkErrorFoundException(ERROR_0)]),
                ])
            ),
            cls(
                find_bugs_side_effect={3: ERROR_0, 4: ERROR_1, 5: ERROR_0, 6: ERROR_1},
                expected=GroupedChunks(error_chunks=[
                    CHUNK_0.replace(lineno=16, end_lineno=20, exceptions=[ChunkErrorFoundException(ERROR_0)]),
                    CHUNK_0.replace(lineno=21, end_lineno=25, exceptions=[ChunkErrorFoundException(ERROR_1)]),
                    CHUNK_0.replace(lineno=26, end_lineno=30, exceptions=[ChunkErrorFoundException(ERROR_0)]),
                    CHUNK_0.replace(lineno=31, end_lineno=35, exceptions=[ChunkErrorFoundException(ERROR_1)]),
                ]),
                die_after=4
            ),
            cls(
                find_bugs_side_effect={3: ERROR_0, 4: ERROR_1, 5: ERROR_0, 6: ERROR_1},
                expected=GroupedChunks(error_chunks=[
                    CHUNK_0.replace(lineno=16, end_lineno=20, exceptions=[ChunkErrorFoundException(ERROR_0)])
                ]),
                die_after=1
            ),
            cls(
                chunks_by_file={'test-0.py': [CHUNK_0], 'test-1.py': [CHUNK_1]},
                find_bugs_side_effect={0: ERROR_0, 1: ERROR_1},
                skip_chunks=set(),
                expected=GroupedChunks(error_chunks=[
                    CHUNK_0.replace(exceptions=[ChunkErrorFoundException(ERROR_0)]),
                    CHUNK_1.replace(exceptions=[ChunkErrorFoundException(ERROR_1)])
                ])
            ),
            cls(
                chunks_by_file={'test-0.py': [CHUNK_0, CHUNK_1]},
                find_bugs_side_effect={0: ERROR_0, 1: ERROR_1},
                skip_chunks={CHUNK_1.get_hash()},
                expected=GroupedChunks(
                    error_chunks=[CHUNK_0.replace(exceptions=[ChunkErrorFoundException(ERROR_0)])],
                    skipped_chunks=[CHUNK_1]
                )
            ),
            cls(
                chunks_by_file={'test-0.py': CHUNK_0_LIST * 10 + CHUNK_SIZE_WARN_LIST},
                expected=GroupedChunks(
                    warning_chunks=[CHUNK_SIZE_WARN.replace(lineno=51, end_lineno=55)]
                )
            ),
            cls(
                chunks_by_file={'test-0.py': CHUNK_0_LIST * 10 + CHUNK_SIZE_ERROR_LIST},
                expected=GroupedChunks(
                    error_chunks=[CHUNK_SIZE_ERROR.replace(lineno=51, end_lineno=55)]
                )
            ),
            cls(
                chunks_by_file={
                    'test-0.py': CHUNK_0_LIST * 10 + CHUNK_PARSE_ERROR_LIST,
                    'test-1.py': CHUNK_1_LIST * 10 + CHUNK_SIZE_WARN_LIST_1 + CHUNK_2_LIST
                },
                find_bugs_side_effect={5: ERROR_0, 23: ERROR_1},
                skip_chunks={CHUNK_2.get_hash()},
                expected=GroupedChunks(
                    error_chunks=[
                        CHUNK_0.replace(lineno=26, end_lineno=30, exceptions=[ChunkErrorFoundException(ERROR_0)]),
                    ],
                    skipped_chunks=[
                        CHUNK_2.replace(lineno=56, end_lineno=60)
                    ],
                    warning_chunks=[
                        CHUNK_PARSE_ERROR.replace(lineno=51, end_lineno=55),
                        CHUNK_SIZE_WARN_1.replace(lineno=51, end_lineno=55)
                    ]
                )
            ),
            cls(
                chunks_by_file={
                    'test-0.py': CHUNK_0_LIST * 10 + CHUNK_PARSE_ERROR_LIST,
                    'test-1.py': CHUNK_1_LIST * 10 + CHUNK_SIZE_WARN_LIST_1 + CHUNK_2_LIST
                },
                find_bugs_side_effect={5: ERROR_0, 23: ERROR_1},
                skip_chunks={CHUNK_2.get_hash()},
                expected=GroupedChunks(
                    error_chunks=[
                        CHUNK_0.replace(lineno=26, end_lineno=30, exceptions=[ChunkErrorFoundException(ERROR_0)]),
                    ],
                    skipped_chunks=[
                        CHUNK_2.replace(lineno=56, end_lineno=60)
                    ],
                    warning_chunks=[
                        CHUNK_PARSE_ERROR.replace(lineno=51, end_lineno=55),
                        CHUNK_SIZE_WARN_1.replace(lineno=51, end_lineno=55)
                    ]
                )
            ),
            cls(
                chunks_by_file={
                    'test-0.py': CHUNK_0_LIST * 10 + CHUNK_PARSE_ERROR_LIST,
                    'test-1.py': CHUNK_1_LIST * 10 + CHUNK_SIZE_WARN_LIST_1 + CHUNK_2_LIST
                },
                find_bugs_side_effect={0: ERROR_0},
                skip_chunks={CHUNK_2.get_hash()},
                line_diffs_by_file={'test-0.py': {30, 31, 51}, 'test-1.py': set(range(45, 60))},
                expected=GroupedChunks(
                    error_chunks=[
                        CHUNK_0.replace(lineno=26, end_lineno=30, exceptions=[ChunkErrorFoundException(ERROR_0)]),
                    ],
                    skipped_chunks=[
                        CHUNK_2.replace(lineno=56, end_lineno=60)
                    ],
                    warning_chunks=[
                        CHUNK_PARSE_ERROR.replace(lineno=51, end_lineno=55),
                        CHUNK_SIZE_WARN_1.replace(lineno=51, end_lineno=55)
                    ]
                )
            ),
            cls(
                chunks_by_file={
                    'test-0.py': CHUNK_0_LIST * 10 + CHUNK_PARSE_ERROR_LIST,
                    'test-1.py': CHUNK_1_LIST * 10 + CHUNK_SIZE_WARN_LIST_1 + CHUNK_2_LIST
                },
                find_bugs_side_effect={1: ERROR_0},
                skip_chunks={CHUNK_2.get_hash()},
                line_diffs_by_file={'test-0.py': {1}},
                expected=GroupedChunks(
                    error_chunks=[
                    ],
                    skipped_chunks=[
                    ],
                    warning_chunks=[
                    ]
                )
            )

        ])


@TestGroupChunksAiParametrize.parametrize()
def test_group_chunks_from_ai(chunks_by_file: Dict[str, List[CodeChunk]], find_bugs_side_effect: Dict[int, str],
                              expected: GroupedChunks, die_after: int, skip_chunks: Set[str],
                              line_diffs_by_file: Optional[Dict[str, Set[int]]]):
    mock_bug_finder = create_autospec(BugFinder, instance=True)
    mock_bug_finder.find_bugs.side_effect = get_find_bugs_side_effect(find_bugs_side_effect)
    grouped_chunks = group_chunks_from_ai(
        bug_finder=mock_bug_finder,
        chunks_by_file=chunks_by_file,
        die_after=die_after,
        line_diffs_by_file=line_diffs_by_file,
        refresh_cache=False,
        skip_chunks=skip_chunks,
    )

    assert grouped_chunks == expected


@pytest.mark.parametrize('code_chunk,line_diffs,expected', [
    (CHUNK_0, {10, 11, 12}, True),
    (CHUNK_0, {1, 100}, False),
    (CHUNK_0.replace(lineno=5, end_lineno=10), {10, 11, 12}, False),
    (CHUNK_0.replace(lineno=5, end_lineno=10), {11, 12}, True),
    (CHUNK_0.replace(lineno=500, end_lineno=501), set(range(400, 600)), False),
    (CHUNK_0.replace(lineno=500, end_lineno=501), set(range(400, 600)) - {500, 501}, True)
])
def test_should_skip_because_not_in_diff(code_chunk: CodeChunk, line_diffs: Set[int], expected: bool):
    actual = should_skip_because_not_in_diff(code_chunk=code_chunk, line_diffs=line_diffs)

    assert actual is expected
