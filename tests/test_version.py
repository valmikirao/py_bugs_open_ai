import os.path
import re
from typing import Optional

import py_bugs_open_ai

CHANGELOG_LINE_RE = re.compile(r'^\*\*([^*]+)\*\*', flags=re.MULTILINE)


def test_changelog(base_dir: str):
    changelog_file = os.path.join(base_dir, 'CHANGELOG.md')
    with open(changelog_file, 'r') as f:
        changelog_text = f.read()
    version: Optional[str] = None
    for version_ in CHANGELOG_LINE_RE.findall(changelog_text):
        version = version_
    assert version == py_bugs_open_ai.__version__
