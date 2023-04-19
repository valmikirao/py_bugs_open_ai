"""Console script for py_bugs_openapi."""
import sys

import click

from .py_bugs_openapi import get_files, ChunkUpCode


@click.command()
@click.argument('path')
def main(path):
    """Console script for py_bugs_openapi."""
    for file in get_files(path):
        with open(file, 'r') as f:
            code = f.read()
        code_chunks = ChunkUpCode(code, max_chunk_size=100).chunks
        divider = '-' * 80
        print()
        print(f"{file}:")
        print(divider)
        for code_chunk in code_chunks:
            print(f"{file}:{code_chunk.lineno} - {code_chunk.end_lineno}")
            print(code_chunk.code)
            print(divider)
    return 0


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
