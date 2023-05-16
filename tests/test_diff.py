import os
import subprocess
import shutil
from tempfile import TemporaryDirectory

from py_bugs_open_ai.diff import get_lines_diffs_by_file
from .constants import BASE_DIR

PY_RESOURCES_DIR = os.path.join(BASE_DIR, 'tests', 'resources', 'test-chunker-params')
DIFF_RESOURCES_DIR = os.path.join(BASE_DIR, 'tests', 'resources', 'diff')


def test_get_lines_diffs_by_file():
    with TemporaryDirectory() as tmp_dir:
        os.chdir(tmp_dir)

        def _cp_from(from_dir: str) -> None:
            files = os.listdir(from_dir)
            for file in files:
                shutil.copy(os.path.join(from_dir, file), os.path.join(tmp_dir, file))
        subprocess.check_call(['git', 'init'])
        _cp_from(PY_RESOURCES_DIR)
        subprocess.check_call(['git', 'add', '.'])
        subprocess.check_call(['git', 'config', '--local', 'user.name', 'Test'])
        subprocess.check_call(['git', 'config', '--local', 'user.email', 'test@test.com'])
        subprocess.check_call(['git', 'commit', '-m', 'TESTING'])
        _cp_from(DIFF_RESOURCES_DIR)
        diff = subprocess.Popen(['git', 'diff'], stdout=subprocess.PIPE)
        actual = dict(get_lines_diffs_by_file(line.decode() for line in diff.stdout))

        try:
            assert actual == {
                'test-1.py': {4, 17, *range(19, 23)},
                'test-long-list.py': {*range(1, 32)},
                'test-multi-peer-groups.py': {13, *range(24, 31)}
            }
        except AssertionError:
            subprocess.check_call(['git', 'diff'])
            raise
