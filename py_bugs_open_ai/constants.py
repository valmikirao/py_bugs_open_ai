import os
from typing import Final

DEFAULT_MODEL: Final[str] = 'gpt-3.5-turbo'
DEFAULT_MAX_CHUNK_SIZE: Final[int] = 500
OPEN_AI_API_KEY: Final[str] = 'OPEN_AI_API_KEY'
_home = os.environ.get('HOME', os.path.join(os.path.abspath(os.sep), 'etc'))
DEFAULT_CACHE: Final[str] = os.path.join(_home, '.pybugsai', 'cache')
DEFAULT_EXAMPLES_FILE: Final[str] = os.path.join(_home, '.pybugsai', 'examples.yml')
DEFAULT_DIE_AFTER: Final[int] = 3
DEFAULT_IS_BUG_RE: Final[str] = r'^ERROR\b'
DEFAULT_MAX_TOKENS_TO_SEND: Final[int] = 1000
FIND_BUGS_SYSTEM_CONTENT = \
    'You are a python bug finder.  Given a snippet of python code, you respond "OK" if you detect no bugs in it' \
    ' and "ERROR: " followed by the error description if you detect an error in it.  Don\'t report import errors' \
    ' packages.'
ERROR_OUT: Final[str] = '\033[91merror\033[0m'  # red "error"
WARN_OUT: Final[str] = '\033[93mwarning\033[0m'  # yellow "warning"
OK_OUT: Final[str] = '\033[92mok\033[0m'  # green "ok"
SKIP_OUT: Final[str] = '\033[93mskip\033[0m'  # yellow "skip"
NOT_IN_DIFF_OUT: Final[str] = '\033[93mnot in diff\033[0m'  # yellow "skip"
EMBEDDINGS_REQUEST_CHUNK_SIZE: Final[int] = 1000
DEFAULT_EMBEDDINGS_MODEL = 'text-embedding-ada-002'
SHORT_DESCRIPTION: Final[str] = 'A utility to help use OpenAI to find bugs in large projects or git diffs in python' \
                                ' code.  Makes heavy use of caching to save time/money'
LICENSE: Final[str] = 'GNU General Public License v3'
CLI_NAME: Final[str] = 'pybugsai'
VERSION: Final[str] = '0.2.0'
AUTHOR: Final[str] = 'Valmiki Rao'
AUTHOR_EMAIL: Final[str] = 'valmikirao@gmail.com'


def get_root_dir() -> str:
    abs_file = os.path.abspath(__file__)
    dirname = os.path.dirname(abs_file)
    root_dir, _ = os.path.split(dirname)
    return root_dir


ROOT_DIR: Final[str] = get_root_dir()
