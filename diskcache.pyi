class Cache:
    def __init__(self, directory: str):
        ...

    # Note: diskcache can store values other than str, but we
    # don't need that here
    def __getitem__(self, item: str) -> str:
        ...

    def __setitem__(self, key: str, value: str) -> None:
        ...

    def __contains__(self, item: str) -> bool:
        ...