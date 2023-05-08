"""Console script for py_bugs_openapi."""
import os
import sys
from typing import MutableMapping

import click
from diskcache import Cache as DiskCache

from py_bugs_open_ai.constants import DEFAULT_MODEL, OPEN_AI_API_KEY, DEFAULT_CODE_CHUNK_SIZE, DEFAULT_CACHE, \
    DEFAULT_DIE_AFTER
from py_bugs_open_ai.py_bugs_open_ai import CodeChunker, BugFinder


@click.command()
@click.option('--api-key-env-variable', default=('%s' % OPEN_AI_API_KEY))
@click.option('--model', default=DEFAULT_MODEL)
@click.option('--code-chunk-size', '--chunk', type=click.INT, default=DEFAULT_CODE_CHUNK_SIZE)
@click.option('--cache-dir', '--cache', default=DEFAULT_CACHE)
@click.option('--refresh-cache', is_flag=True)
@click.option('--die-after', type=click.INT, default=DEFAULT_DIE_AFTER)
@click.option('--verbose', is_flag=True)
def main(api_key_env_variable: str, model: str, code_chunk_size: int, cache_dir: str, refresh_cache: bool,
         die_after: int, verbose: bool) -> int:
    """Console script for py_bugs_openapi."""
    api_key = os.environ[api_key_env_variable]
    os.makedirs(cache_dir, exist_ok=True)
    cache = DiskCache(cache_dir)
    bug_finder = BugFinder(model=model, api_key=api_key, cache=cache)
    error_count = 0

    for line in sys.stdin:
        file = line.strip('\n')
        with open(file, 'r') as f:
            code = f.read()

        divider = '-' * 80
        print()
        print(f"{file}:")
        print(divider)

        code_chunks = CodeChunker(code, max_chunk_size=code_chunk_size, model=model).get_chunks()
        for code_chunk in code_chunks:
            print(f"{file}:{code_chunk.lineno} - {code_chunk.end_lineno}; {code_chunk.token_count}")

            has_bugs, bugs_description = bug_finder.find_bugs(code_chunk.code, refresh_cache=refresh_cache)

            if has_bugs:
                error_count += 1
                print(divider)
                print(code_chunk.code)
                print(divider)
                print(bugs_description, file=sys.stderr)
                print(divider)
            else:
                if verbose:
                    code_lines = code_chunk.code.split('\n')
                    for code_line in code_lines[:10]:
                        print(code_line)
                    if len(code_lines) > 10:
                        print('[...]')
                    print(divider)
                    print(bugs_description)
                    print(divider)

            if error_count >= die_after:
                print(f'Dying after {error_count} errors', file=sys.stderr)
                sys.exit(1)


    if error_count > 0:
        print(f'{error_count} errors found', file=sys.stderr)
        sys.exit(1)
    else:
        print('No errors found')
        return 0


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
