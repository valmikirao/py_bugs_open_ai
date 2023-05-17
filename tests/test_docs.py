import os.path

from py_bugs_open_ai.constants import ROOT_DIR
from scripts.generate_readme import render_readme


def test_docs():
    """ Just make sure the README.md has been properly generated"""
    rendered_readme = render_readme()
    readme_path = os.path.join(ROOT_DIR, 'README.md')
    with open(readme_path, 'r') as f:
        readme_text = f.read()

    assert rendered_readme == readme_text
