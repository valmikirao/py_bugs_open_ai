from typing import Optional, TypeVar


def assert_strict(is_true: bool, message: str = 'ERROR') -> None:
    """
    Assert that fails regardless of whether production key is set
    """
    if not is_true:
        raise AssertionError(message)


T = TypeVar('T')


def coalesce(*args: Optional[T]) -> T:
    for arg in args:
        if arg is not None:
            return arg
    raise TypeError('At least one argument needs to not be None')
