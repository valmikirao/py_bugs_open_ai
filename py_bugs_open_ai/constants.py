import os
from typing import Final

DEFAULT_MODEL: Final[str] = 'gpt-3.5-turbo'
DEFAULT_CODE_CHUNK_SIZE: Final[int] = 500
OPEN_AI_API_KEY: Final[str] = 'OPEN_AI_API_KEY'
_home = os.environ.get('HOME', os.path.join(os.path.abspath(os.sep), 'etc'))
DEFAULT_CACHE: Final[str] = os.path.join(_home, '.pybugs/cache')
DEFAULT_DIE_AFTER: Final[int] = 3