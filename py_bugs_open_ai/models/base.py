import json
from typing import Dict, Any, Protocol, Generic, TypeVar

from pydantic import BaseModel
from diskcache import Cache

class MyBaseModel(BaseModel):

    def full_dict(self) -> Dict[str, Any]:
        return json.loads(self.json())


K = TypeVar('K')
V = TypeVar('V')


class CacheProtocol(Protocol[K, V]):
    def __getitem__(self, item: K) -> V:
        ...

    def __setitem__(self, key: K, value: V) -> None:
        ...

    def __contains__(self, item: V) -> bool:
        ...
