"""Console script for py_bugs_openapi."""
import itertools
import os
import re
import sys
from collections import OrderedDict
from configparser import ConfigParser
from dataclasses import dataclass, field
from typing import List, Callable, Set, Iterator, Tuple, Optional, Dict, Iterable, cast, TypeVar, Protocol

import click
import yaml
from diskcache import Cache as DiskCache  # type: ignore

from py_bugs_open_ai.constants import DEFAULT_MODEL, OPEN_AI_API_KEY, DEFAULT_MAX_CHUNK_SIZE, DEFAULT_CACHE, \
    DEFAULT_DIE_AFTER, ERROR_OUT, WARN_OUT, OK_OUT, CLI_NAME, SKIP_OUT, DEFAULT_EXAMPLES_FILE, \
    DEFAULT_EMBEDDINGS_MODEL, DEFAULT_IS_BUG_RE, FIND_BUGS_SYSTEM_CONTENT, NOT_IN_DIFF_OUT, DEFAULT_MAX_TOKENS_TO_SEND
from py_bugs_open_ai.diff import get_lines_diffs_by_file
from py_bugs_open_ai.models.base import CacheProtocol
from py_bugs_open_ai.models.examples import ExamplesFile
from py_bugs_open_ai.open_ai_client import OpenAiClient
from py_bugs_open_ai.py_bugs_open_ai import CodeChunker, BugFinder, CodeChunk, QueryConstructor, \
    ChunkErrorFoundException

DEFAULT_CONFIG_FILES = [
    'pybugsai.cfg',
    'setup.cfg'
]

K = TypeVar('K')
V = TypeVar('V')


def _query_cache_wrapper(directory: str) -> CacheProtocol[str, str]:
    # wrapper to correctly type the query cache's DiskCache
    return cast(CacheProtocol[str, str], DiskCache(directory))


def _embeddings_cache_wrapper(directory: str) -> CacheProtocol[str, List[float]]:
    # wrapper to correctly type the embeddings cache's DiskCache
    return cast(CacheProtocol[str, List[float]], DiskCache(directory))


def _handle_config(ctx: click.Context, param: click.Option, filename: Optional[str]) -> Optional[str]:
    cfg = ConfigParser()
    if filename is not None:
        cfg.read(filename)
    else:
        for default_config_file in DEFAULT_CONFIG_FILES:
            cfg.read(default_config_file)
    if CLI_NAME in cfg:
        ctx.default_map = dict(cfg[CLI_NAME])
        for multi_param in ('file', 'skip_chunks'):
            if multi_param in ctx.default_map:
                ctx.default_map[multi_param] = [
                    c for c in re.split(r'\s+', ctx.default_map[multi_param]) if c != ''
                ]

    return filename


def _handle_skip_chunks(ctx: click.Context, param: click.Option, skip_chunks: Tuple[str]) -> Set[str]:
    split_skip_chunks = (c.split(',') for c in skip_chunks)
    flattened_split_skip_chunks = itertools.chain(*split_skip_chunks)
    return set(flattened_split_skip_chunks)


def get_bug_finder_cache_dir(cache_dir: str) -> str:
    return os.path.join(cache_dir, 'bug_finder')


def get_embeddings_cache_dir(cache_dir: str) -> str:
    return os.path.join(cache_dir, 'embeddings')


def prime_embeddings_cache(chunks: Iterable[CodeChunk], open_ai_client: OpenAiClient,
                           query_constructor: QueryConstructor):
    chunks_that_need_embeddings: List[CodeChunk] = []
    chunk: CodeChunk
    for chunk in chunks:
        if query_constructor.will_filter_examples(chunk.code):
            chunks_that_need_embeddings.append(chunk)
    if len(chunks_that_need_embeddings) > 0:
        texts_iter: Iterator[str] = itertools.chain(
            (c.code for c in chunks_that_need_embeddings),
            (e.code for e in query_constructor.examples)
        )
        open_ai_client.get_embeddings(texts=texts_iter)


def should_skip_because_not_in_diff(code_chunk: CodeChunk, line_diffs: Set[int]):
    if code_chunk.end_lineno == 30 and code_chunk.file == 'test-0.py':
        pass
    chunk_linenos = range(code_chunk.lineno, code_chunk.end_lineno + 1)
    if any(lineno in line_diffs for lineno in chunk_linenos):
        skip_because_not_in_diff = False
    else:
        skip_because_not_in_diff = True
    return skip_because_not_in_diff


def _color_func(color: str) -> Callable[[str], str]:
    def _func(message: str) -> str:
        return f"{color}{message}\033[0m"

    return _func


_red = _color_func('\033[91m')
_yellow = _color_func('\033[93m')
_green = _color_func('\033[92m')


def _main(abs_max_chunk_size: int, api_key: str, cache_dir: str, die_after: int, diff_from_stdin: bool,
          embeddings_model: str, examples_file: Optional[str], file: List[str], files_from_stdin: bool,
          is_bug_re: re.Pattern, max_chunk_size: int, max_tokens_to_send: int, model: str, refresh_cache: bool,
          skip_chunks: Set[str], strict_chunk_size: bool, system_content: str) -> None:
    query_cache = _query_cache_wrapper(get_bug_finder_cache_dir(cache_dir))
    embeddings_cache = _embeddings_cache_wrapper(get_embeddings_cache_dir(cache_dir))
    open_ai_client = OpenAiClient(
        api_key=api_key,
        model=model,
        embedding_model=embeddings_model,
        query_cache=query_cache,
        embeddings_cache=embeddings_cache,
    )
    bug_finder = BugFinder(
        open_ai_client=open_ai_client,
        is_bug_re=is_bug_re,
        system_content=system_content
    )
    query_constructor: Optional[QueryConstructor]
    if examples_file:
        with open(examples_file, 'r') as f:
            examples_obj = yaml.full_load(f)
        examples_file_ = ExamplesFile.parse_obj(examples_obj)

        query_constructor = QueryConstructor(
            open_ai_client=open_ai_client,
            max_tokens_to_send=max_tokens_to_send,
            system_content=system_content,
            model=model,
            examples=examples_file_.examples
        )
    else:
        query_constructor = None

    file_list: List[str]
    line_diffs_by_file: Optional[Dict[str, Set[int]]]
    if files_from_stdin:
        line_diffs_by_file = None
        file_list = [line.strip('\n') for line in sys.stdin]
    elif diff_from_stdin:
        line_diffs_by_file = get_lines_diffs_by_file(sys.stdin)
        file_list = sorted(line_diffs_by_file.keys())
    else:
        line_diffs_by_file = None
        file_list = file
    chunks_by_file: Dict[str, List[CodeChunk]] = OrderedDict()

    for file_ in file_list:
        if os.path.exists(file_):
            with open(file_, 'r') as f:
                code = f.read()

            chunks_by_file[file_] = list(CodeChunker(
                code,
                file=file_,
                max_chunk_size=max_chunk_size,
                model=model,
                abs_max_chunk_size=abs_max_chunk_size,
                strict_chunk_size=strict_chunk_size
            ).get_chunks())
    # prime the embeddings cache if needed
    if query_constructor is not None:
        all_chunks = itertools.chain(*chunks_by_file.values())
        prime_embeddings_cache(
            chunks=all_chunks,
            open_ai_client=open_ai_client,
            query_constructor=query_constructor
        )

    def _echo(message: str, nl: bool = True, err: bool = False) -> None:
        click.echo(message, nl=nl, err=err)

    grouped_chunks = group_chunks_from_ai(
        bug_finder=bug_finder,
        chunks_by_file=chunks_by_file,
        die_after=die_after,
        line_diffs_by_file=line_diffs_by_file,
        refresh_cache=refresh_cache,
        skip_chunks=skip_chunks,
        echo=_echo
    )

    error_chunks = grouped_chunks.error_chunks
    skipped_chunks = grouped_chunks.skipped_chunks
    warning_chunks = grouped_chunks.warning_chunks

    divider = '-' * 80

    def _summary_formatter(chunks: List[CodeChunk], color_func: Callable[[str], str], header_template: str):
        if len(chunks) > 0:
            click.echo(color_func(divider), file=sys.stderr)
            header = header_template.format(count=len(chunks))
            click.echo(color_func(header), file=sys.stderr)
            for chunk in chunks:
                exception_messages = '; '.join(e.message for e in chunk.exceptions)
                click.echo(color_func(f"{_chunk_header(chunk)} - {exception_messages}"))
            click.echo(color_func(divider), file=sys.stderr)

    _summary_formatter(warning_chunks, _yellow, '{count} warnings')
    _summary_formatter(skipped_chunks, _yellow, '{count} skipper')
    _summary_formatter(error_chunks, _red, '{count} errors')

    if len(error_chunks) > 0:
        sys.exit(1)
    else:
        click.echo(_green('No errors found'))


class EchoFunc(Protocol):
    def __call__(self, message: str, nl: bool = True, err: bool = False) -> None:
        ...


def echo_func_noop(message: str, nl: bool = True, err: bool = False) -> None:
    pass


def _chunk_header(chunk: CodeChunk) -> str:
    return f"{chunk.file}:{chunk.lineno}-{chunk.end_lineno};" \
           f" {chunk.get_hash()} token count: {chunk.token_count}"


@dataclass
class GroupedChunks:
    error_chunks: List[CodeChunk] = field(default_factory=lambda: [])
    warning_chunks: List[CodeChunk] = field(default_factory=lambda: [])
    skipped_chunks: List[CodeChunk] = field(default_factory=lambda: [])


def group_chunks_from_ai(bug_finder: BugFinder, chunks_by_file: Dict[str, List[CodeChunk]], die_after: int,
                         line_diffs_by_file: Optional[Dict[str, Set[int]]],
                         refresh_cache: bool, skip_chunks: Set[str], echo: EchoFunc = echo_func_noop) \
        -> GroupedChunks:

    error_chunks: List[CodeChunk] = []
    warning_chunks: List[CodeChunk] = []
    skipped_chunks: List[CodeChunk] = []
    chunks_by_file_items = list(chunks_by_file.items())
    for file_, code_chunks in chunks_by_file_items:
        for code_chunk in code_chunks:
            echo(f"{_chunk_header(code_chunk)} - ", nl=False)

            if line_diffs_by_file is not None:
                skip_because_not_in_diff = should_skip_because_not_in_diff(
                    code_chunk=code_chunk,
                    line_diffs=line_diffs_by_file.get(file_, set())
                )
            else:
                skip_because_not_in_diff = False

            if not skip_because_not_in_diff and not code_chunk.has_exception() \
                    and code_chunk.get_hash() not in skip_chunks:
                has_bugs, bugs_description = bug_finder.find_bugs(code_chunk.code, refresh_cache=refresh_cache)
                if has_bugs:
                    code_chunk.exceptions.append(ChunkErrorFoundException(bugs_description))

            code_chunk_hash = code_chunk.get_hash()
            if not skip_because_not_in_diff and code_chunk.has_exception(is_error=True) and \
                    code_chunk_hash not in skip_chunks:
                echo(ERROR_OUT)
                error_chunks.append(code_chunk)
            elif not skip_because_not_in_diff and code_chunk_hash in skip_chunks:
                echo(SKIP_OUT)
                skipped_chunks.append(code_chunk)
            elif not skip_because_not_in_diff and code_chunk.has_exception(is_error=False):
                echo(WARN_OUT)
                warning_chunks.append(code_chunk)
            elif skip_because_not_in_diff:
                echo(NOT_IN_DIFF_OUT)
            else:
                echo(OK_OUT)

            if len(error_chunks) >= die_after:
                echo(_red(f'Dying after {len(error_chunks)} errors'), err=True)
                break
        if len(error_chunks) >= die_after:
            break

    return GroupedChunks(
        error_chunks=error_chunks,
        skipped_chunks=skipped_chunks,
        warning_chunks=warning_chunks
    )


def _file_for_help(file: str) -> str:
    if 'HOME' in os.environ:
        pattern = '^' + re.escape(os.environ['HOME'])
        return re.sub(pattern, '~', file)
    else:
        return file


@click.command()
@click.argument('file', nargs=-1)
@click.option('--config', '-c', callback=_handle_config,
              help=f"The config file.  Overrides the [pybugsai] section in {DEFAULT_CONFIG_FILES[0]} and"
                   f" {DEFAULT_CONFIG_FILES[1]}")
@click.option('--files-from-stdin', '--in', is_flag=True,
              help='Take the list of files from standard in, such that you could run this script like'
                   ' `git ls-files -- \'*.py\' | pybugsai --in`')
@click.option('--api-key-env-variable', default=OPEN_AI_API_KEY, show_default=True,
              help='The environment variable which the openai api key is stored in')
@click.option('--model', default=DEFAULT_MODEL, show_default=True, help='The openai model used')
@click.option('--embeddings-model', default=DEFAULT_EMBEDDINGS_MODEL)
@click.option('--max-chunk-size', '--chunk', type=click.INT, default=DEFAULT_MAX_CHUNK_SIZE, show_default=True,
              help='The script tries to break the python down into chunk sizes smaller than this')
@click.option('--abs-max-chunk-size', '--abs-chunk', type=click.INT,
              help='Sometimes the script can\'t break up the code into chunks smaller than --max-chunk-size.  This'
                   ' is the absolute maximum size of chunk it will send.  If a chunk is bigger than this, it will be'
                   ' reported as a warning or as an error if --strict-chunk-size is set.  Defaults to --max-chunk-size')
@click.option('--cache-dir', '--cache', default=DEFAULT_CACHE,
              help=f'The cache directory [{_file_for_help(DEFAULT_CACHE)}]')
@click.option('--refresh-cache', is_flag=True)
@click.option('--die-after', type=click.INT, default=DEFAULT_DIE_AFTER, show_default=True,
              help='After this many errors are found, the scripts stops running')
@click.option('--strict-chunk-size', '--strict', is_flag=True,
              help='If true and there is a chunk that is bigger than --abs-max-chunk-size, it will be marked as an'
                   ' error')
@click.option('--skip-chunks', multiple=True, callback=_handle_skip_chunks,
              help='The hashes of the chunks to skip.  Can be added multiple times are be a comma-delimited list')
@click.option('--diff-from-stdin', '--diff-in', is_flag=True,
              help='Be able to take `git diff` from the std-in and then only check the chunks for lines that are'
                   ' different')
@click.option('--is-bug-re', '--re', default=DEFAULT_IS_BUG_RE, show_default=True,
              help='If the response from OpenAI matches this regular-expression, then it is marked as an error.  Might'
                   ' be necessary to change this from the default if you use a customer --system-content')
@click.option('--is-bug-re-ignore-case', '-i', is_flag=True, help='Ignore the case when applying the `--is-bug-re`')
@click.option('--system-content', '-s', default=FIND_BUGS_SYSTEM_CONTENT, show_default=False,
              help='The system content sent to OpenAI')
@click.option('--examples-file', default=DEFAULT_EXAMPLES_FILE,
              help='File containing example code and responses to guide openai in finding bugs or non-bugs.  See'
                   f' README for format and more information [default: {_file_for_help(DEFAULT_EXAMPLES_FILE)}]')
@click.option('--max-tokens-to-send', type=click.INT, default=DEFAULT_MAX_TOKENS_TO_SEND, show_default=True,
              help='Maximum number of tokens to send to the OpenAI api, include the examples in the --examples-file.'
                   f'  {CLI_NAME} uses embeddings to only send the most relevant examples if it can\'t send them all'
                   f' without exceeding this count')
def main(file: List[str], files_from_stdin: bool, api_key_env_variable: str, model: str, embeddings_model: str,
         max_chunk_size: int, abs_max_chunk_size: Optional[int], cache_dir: str, refresh_cache: bool, die_after: int,
         strict_chunk_size: bool, config: str, skip_chunks: Set[str], diff_from_stdin, is_bug_re: str,
         is_bug_re_ignore_case: bool, system_content: str, max_tokens_to_send: int, examples_file: str) -> int:
    """Chunks up python files and sends the pieces to open-ai to see if it thinks there are any bugs in it"""

    api_key = os.environ[api_key_env_variable]
    os.makedirs(cache_dir, exist_ok=True)

    if is_bug_re_ignore_case:
        is_bug_re_flags = re.IGNORECASE
    else:
        is_bug_re_flags = cast(re.RegexFlag, 0)
    is_bug_re_ = re.compile(is_bug_re, flags=is_bug_re_flags)

    abs_max_chunk_size_ = abs_max_chunk_size if abs_max_chunk_size is not None else max_chunk_size
    if abs_max_chunk_size_ < max_chunk_size:
        raise click.BadOptionUsage(
            option_name='abs-max-chunk-size',
            message=f'--abs-max-chunk-size ({abs_max_chunk_size_}) cannot be less than --max-chunk-size'
                    f' ({max_chunk_size})'
        )
    examples_file_: Optional[str]
    if os.path.exists(examples_file):  # module os imported above (pybugsai)
        examples_file_ = examples_file
    else:
        examples_file_ = None
        if examples_file != DEFAULT_EXAMPLES_FILE:
            click.echo(f"WARNING: Examples file {examples_file!r}", err=True)

    _main(
        abs_max_chunk_size=abs_max_chunk_size_,
        api_key=api_key,
        cache_dir=cache_dir,
        die_after=die_after,
        diff_from_stdin=diff_from_stdin,
        embeddings_model=embeddings_model,
        examples_file=examples_file_,
        file=file,
        files_from_stdin=files_from_stdin,
        is_bug_re=is_bug_re_,
        max_chunk_size=max_chunk_size,
        max_tokens_to_send=max_tokens_to_send,
        model=model,
        refresh_cache=refresh_cache,
        skip_chunks=skip_chunks,
        strict_chunk_size=strict_chunk_size,
        system_content=system_content
    )

    return 0


if __name__ == "__main__":
    # sys imported above (pybugsai)
    sys.exit(main())  # pragma: no cover
