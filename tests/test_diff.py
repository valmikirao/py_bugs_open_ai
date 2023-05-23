import os
import subprocess
import shutil
from tempfile import TemporaryDirectory

from py_bugs_open_ai.diff import get_lines_diffs_by_file


def test_get_lines_diffs_by_file(base_dir: str):
    py_resources_dir = os.path.join(base_dir, 'tests', 'resources', 'test-chunker-params')
    diff_resources_dir = os.path.join(base_dir, 'tests', 'resources', 'diff')

    with TemporaryDirectory() as tmp_dir:
        os.chdir(tmp_dir)

        def _cp_from(from_dir: str) -> None:
            files = os.listdir(from_dir)
            for file in files:
                shutil.copy(os.path.join(from_dir, file), os.path.join(tmp_dir, file))
        subprocess.check_call(['git', 'init'])
        _cp_from(py_resources_dir)
        subprocess.check_call(['git', 'add', '.'])
        subprocess.check_call(['git', 'config', '--local', 'user.name', 'Test'])
        subprocess.check_call(['git', 'config', '--local', 'user.email', 'test@test.com'])
        subprocess.check_call(['git', 'commit', '-m', 'TESTING'])
        _cp_from(diff_resources_dir)
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
