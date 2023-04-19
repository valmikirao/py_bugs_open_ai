"""Main module."""
import ast
import os
from dataclasses import dataclass
from glob import iglob
from typing import Iterable, List, Optional, Any
from uuid import uuid4, UUID


def get_files(path: str = '.', glob: str = '*.py') -> Iterable[str]:
    if os.path.isfile(path):
        yield path
    else:
        for dir, _, _ in os.walk(path):
            for base_name in iglob(glob, root_dir=dir):
                yield os.path.join(dir, base_name)

@dataclass
class CodeChunk:
    lineno: int
    end_lineno: int
    code: str
    peer_group: UUID


class ChunkUpCode(ast.NodeVisitor):
    def __init__(self, code: str, max_chunk_size: int):
        self.chunks: List[CodeChunk] = []
        self.max_chunk_size = max_chunk_size
        self._current_peer_group = uuid4()
        self._code_lines = code.split('\n')
        self._tree = ast.parse(code)

        self.visit(self._tree)

    def chunk_from_node(self, node: ast.AST) -> Optional[CodeChunk]:
        if hasattr(node, 'lineno') and hasattr(node, 'end_lineno'):
            lines = self._code_lines[node.lineno-1:node.end_lineno]
            col_offset = getattr(node, 'col_offset', 0)
            # lines = [l[col_offset:] for l in lines]
            code = '\n'.join(lines)

            return CodeChunk(
                lineno=node.lineno,
                end_lineno=node.end_lineno,
                code=code,
                peer_group=self._current_peer_group
            )

    def generic_visit(self, node: ast.AST) -> Any:
        chunk = self.chunk_from_node(node)
        if chunk is not None \
                and len(chunk.code) <= self.max_chunk_size:
            last_chunk = self.chunks[-1] if self.chunks else None
            if last_chunk is not None \
                    and last_chunk.peer_group == chunk.peer_group \
                    and len(chunk.code) + len(last_chunk.code) + 1 <= self.max_chunk_size:
                self.chunks[-1] = CodeChunk(
                    lineno=last_chunk.lineno,
                    end_lineno=chunk.end_lineno,
                    code=last_chunk.code + '\n' + chunk.code,
                    peer_group=chunk.peer_group
                )
            else:
                self.chunks.append(chunk)
        else:
            old_peer_group = self._current_peer_group
            try:
                self._current_peer_group = uuid4()
                super().generic_visit(node)
            finally:
                self._current_peer_group = old_peer_group
