import os
import re
from collections import defaultdict
from typing import Iterable, MutableMapping, Set


REMOVED_FILE_NAME_PREFIX = f'--- a{os.path.sep}'
FILE_NAME_PREFIX = f'+++ b{os.path.sep}'


def get_lines_diffs_by_file(in_lines: Iterable[str]) -> MutableMapping[str, Set[int]]:
    file = ''
    added_lineno = 0
    removed_lineno = 0
    line_diffs_by_file: MutableMapping[str, Set[int]] = defaultdict(set)
    for line in in_lines:
        if line.startswith(FILE_NAME_PREFIX):
            match = re.search(r'^' + re.escape(FILE_NAME_PREFIX) + r'([^\t\n\r]*)', line)
            assert match, 'This should match, is this a valid git diff?'
            file = match.group(1)
        elif line.startswith('@@'):
            if added_match := re.search(r'\+(\d+)', line):
                added_lineno = int(added_match.group(1)) - 1
            else:
                added_lineno = -1
            if removed_match := re.search(r'-(\d+)', line):
                removed_lineno = int(removed_match.group(1)) - 1
            else:
                removed_lineno = - 1
        elif line.startswith('+') and not line.startswith(FILE_NAME_PREFIX):
            added_lineno += 1
            line_diffs_by_file[file].add(added_lineno)
        elif line.startswith('-') and not line.startswith(REMOVED_FILE_NAME_PREFIX):
            removed_lineno += 1
            line_diffs_by_file[file].add(removed_lineno)
        elif line.startswith(' '):
            added_lineno += 1
            removed_lineno += 1
        else:
            pass

    return line_diffs_by_file
