import os
from typing import Final

DEFAULT_MODEL: Final[str] = 'gpt-3.5-turbo'
DEFAULT_MAX_CHUNK_SIZE: Final[int] = 500
OPEN_AI_API_KEY: Final[str] = 'OPEN_AI_API_KEY'
_home = os.environ.get('HOME', os.path.join(os.path.abspath(os.sep), 'etc'))
PYBUGSAI_DIR = os.path.join(_home, '.pybugsai')
DEFAULT_CACHE: Final[str] = os.path.join(PYBUGSAI_DIR, 'cache')
DEFAULT_EXAMPLE_FILE: Final[str] = os.path.join(PYBUGSAI_DIR, 'examples.yml')
DEFAULT_DIE_AFTER: Final[int] = 3
ERROR_OUT: Final[str] = '\033[91merror\033[0m'  # red "error"
WARN_OUT: Final[str] = '\033[93mwarning\033[0m'  # yellow "warning"
OK_OUT: Final[str] = '\033[92mok\033[0m'  # green "ok"
SKIP_OUT: Final[str] = '\033[93mskip\033[0m'  # yellow "skip"
CLI_NAME: Final[str] = 'pybugsai'
EMBEDDINGS_REQUEST_CHUNK_SIZE: Final[int] = 1000
DEFAULT_EMBEDDING_MODEL = 'text-embedding-ada-002'
