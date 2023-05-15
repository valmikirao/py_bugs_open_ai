"""Console script for py_bugs_openapi."""
import itertools
import os
import re
import sys
from configparser import ConfigParser
from typing import List, Callable, Set, Iterable, Tuple, MutableMapping

import click
from diskcache import Cache as DiskCache

from py_bugs_open_ai.constants import DEFAULT_MODEL, OPEN_AI_API_KEY, DEFAULT_MAX_CHUNK_SIZE, DEFAULT_CACHE, \
    DEFAULT_DIE_AFTER, ERROR_OUT, WARN_OUT, OK_OUT, CLI_NAME, SKIP_OUT
from py_bugs_open_ai.diff import get_lines_diffs_by_file
from py_bugs_open_ai.py_bugs_open_ai import CodeChunker, BugFinder, CodeChunk


class StdInIterable:
    def __next__(self):
        line = next(sys.stdin)
        return line.strip('\n')

    def __iter__(self):
        return self


DEFAULT_CONFIG_FILES = [
    'pybugsai.cfg',
    'setup.cfg'
]


def _handle_config(ctx: click.Context, param: click.Option, filename: str | None) -> str | None:
    cfg = ConfigParser()
    if filename is not None:
        cfg.read(filename)
    else:
        for default_config_file in DEFAULT_CONFIG_FILES:
            cfg.read(default_config_file)
    if CLI_NAME in cfg:
        ctx.default_map = dict(cfg[CLI_NAME])
        if 'skip_chunks' in ctx.default_map:
            ctx.default_map['skip_chunks'] = [c for c in re.split(r'\s+', ctx.default_map['skip_chunks']) if c != '']

    return filename


def _handle_skip_chunks(ctx: click.Context, param: click.Option, skip_chunks: Tuple[str]) -> Set[str]:
    split_skip_chunks = (c.split(',') for c in skip_chunks)
    flattened_split_skip_chunks = itertools.chain(*split_skip_chunks)
    return set(flattened_split_skip_chunks)


@click.command()
@click.argument('file', nargs=-1)
@click.option('--config', '-c', callback=_handle_config)
@click.option('--files-from-stdin', '--in', is_flag=True)
@click.option('--api-key-env-variable', default=('%s' % OPEN_AI_API_KEY))
@click.option('--model', default=DEFAULT_MODEL)
@click.option('--max-chunk-size', '--chunk', type=click.INT, default=DEFAULT_MAX_CHUNK_SIZE)
@click.option('--abs-max-chunk-size', '--abs-chunk', type=click.INT, default=DEFAULT_MAX_CHUNK_SIZE)
@click.option('--cache-dir', '--cache', default=DEFAULT_CACHE)
@click.option('--refresh-cache', is_flag=True)
@click.option('--die-after', type=click.INT, default=DEFAULT_DIE_AFTER)
@click.option('--strict-chunk-size', '--strict', is_flag=True)
@click.option('--skip-chunks', multiple=True, callback=_handle_skip_chunks)
@click.option('--diff-from-stdin', '--diff-in', is_flag=True)
def main(file: List[str], files_from_stdin: bool, api_key_env_variable: str, model: str, max_chunk_size: int,
         abs_max_chunk_size: int, cache_dir: str, refresh_cache: bool, die_after: int,
         strict_chunk_size: bool, config: str, skip_chunks: Set[str], diff_from_stdin) -> int:
    """Console script for py_bugs_openapi."""
    api_key = os.environ[api_key_env_variable]
    os.makedirs(cache_dir, exist_ok=True)
    cache = DiskCache(cache_dir)
    bug_finder = BugFinder(model=model, api_key=api_key, cache=cache)

    def _color_func(color: str) -> Callable[[str], str]:
        def _func(message: str) -> str:
            return f"{color}{message}\033[0m"
        return _func

    _red = _color_func('\033[91m')
    _yellow = _color_func('\033[93m')
    _green = _color_func('\033[92m')

    def _chunk_header(chunk: CodeChunk) -> str:
        return f"{chunk.file}:{code_chunk.lineno}-{code_chunk.end_lineno};" \
               f" {chunk.get_hash()} token count: {code_chunk.token_count}"

    error_chunks: List[CodeChunk] = []
    warning_chunks: List[CodeChunk] = []

    file_iterable: Iterable[str]
    line_diffs_by_file: MutableMapping[str, Set[int]] = {}
    if files_from_stdin:
        file_iterable = StdInIterable()
    elif diff_from_stdin:
        line_diffs_by_file = get_lines_diffs_by_file(sys.stdin)
        file_iterable = sorted(line_diffs_by_file.keys())
    else:
        file_iterable = file

    for file_ in file_iterable:
        with open(file_, 'r') as f:
            code = f.read()

        code_chunks = CodeChunker(
            code, file=file_, max_chunk_size=max_chunk_size, model=model, abs_max_chunk_size=abs_max_chunk_size,
            strict_chunk_size=strict_chunk_size
        ).get_chunks()
        for code_chunk in code_chunks:
            click.echo(f"{_chunk_header(code_chunk)} - ", nl=False)

            if diff_from_stdin:
                chunk_linenos = range(code_chunk.lineno, code_chunk.end_lineno)
                if any(lineno in line_diffs_by_file[file_] for lineno in chunk_linenos):
                    skip_because_not_in_diff = False
                else:
                    skip_because_not_in_diff = True
            else:
                skip_because_not_in_diff = False

            if not skip_because_not_in_diff and code_chunk.error is None and code_chunk.warning is None:
                has_bugs, bugs_description = bug_finder.find_bugs(code_chunk.code, refresh_cache=refresh_cache)
                if has_bugs:
                    code_chunk.error = bugs_description
            code_chunk_hash = code_chunk.get_hash()
            if not skip_because_not_in_diff and code_chunk.error is not None and code_chunk_hash not in skip_chunks:
                click.echo(ERROR_OUT)
                error_chunks.append(code_chunk)
            if not skip_because_not_in_diff and code_chunk.error is not None and code_chunk_hash in skip_chunks:
                click.echo(SKIP_OUT)
                code_chunk.warning = 'SKIPPED: ' + code_chunk.error
                code_chunk.error = None
                warning_chunks.append(code_chunk)
            elif not skip_because_not_in_diff and code_chunk.warning is not None:
                click.echo(WARN_OUT)
                warning_chunks.append(code_chunk)
            elif skip_because_not_in_diff:
                click.echo(SKIP_OUT)
            else:
                click.echo(OK_OUT)

            if len(error_chunks) >= die_after:
                click.echo(_red(f'Dying after {len(error_chunks)} errors'), err=True)
                break
        if len(error_chunks) >= die_after:
            break

    divider = '-' * 80

    if len(warning_chunks) > 0:
        click.echo(_yellow(divider), file=sys.stderr)
        click.echo(_yellow(f"{len(warning_chunks)} warnings"), file=sys.stderr)
        for chunk in warning_chunks:
            click.echo(_yellow(f"{_chunk_header(chunk)} - {chunk.warning}"))
        click.echo(_yellow(divider), file=sys.stderr)
    if len(error_chunks) > 0:
        click.echo(_red(divider), file=sys.stderr)
        click.echo(_red(f'{len(error_chunks)} errors found'), file=sys.stderr)
        for chunk in error_chunks:
            click.echo(_red(f"{_chunk_header(chunk)} - {chunk.error}"))
        click.echo(_red(divider), file=sys.stderr)

        sys.exit(1)
    else:
        click.echo(_green('No errors found'))
        return 0


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
