from typing import Iterable, Tuple


class AST:
    ...


class stmt(AST):
    lineno: int
    end_lineno: int
    col_offset: int
    end_col_offset: int


class NodeVisitor:
    def visit(self, node: AST) -> None:
        ...

    def generic_visit(self, node: AST) -> None:
        ...


def parse(source: str) -> AST:
    ...

def iter_fields(node: AST) -> Iterable[Tuple[str, AST]]:
    ...

def iter_child_nodes(node: AST) -> Iterable[AST]:
    ...