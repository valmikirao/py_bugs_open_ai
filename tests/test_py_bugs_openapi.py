#!/usr/bin/env python

"""Tests for `py_bugs_open_ai` package."""

import pytest

from click.testing import CliRunner

from py_bugs_open_ai import cli
from py_bugs_open_ai.py_bugs_open_ai import CodeChunker


@pytest.fixture
def response():
    """Sample pytest fixture.

    See more at: http://doc.pytest.org/en/latest/fixture.html
    """
    # import requests
    # return requests.get('https://github.com/audreyr/cookiecutter-pypackage')


def test_content(response):
    """Sample pytest test function with the pytest fixture as an argument."""
    # from bs4 import BeautifulSoup
    # assert 'GitHub' in BeautifulSoup(response.content).title.string


def test_command_line_interface():
    """Test the CLI."""
    runner = CliRunner()
    result = runner.invoke(cli.main)
    assert result.exit_code == 0
    assert 'py_bugs_open_ai.cli.main' in result.output
    help_result = runner.invoke(cli.main, ['--help'])
    assert help_result.exit_code == 0
    assert '--help  Show this message and exit.' in help_result.output


@pytest.mark.parametrize("total_token_count,max_chunk_size,expected", [
    (100, 10, 10),
    (100, 50, 25),
    (100, 100, 50),
    (100, 200, 100),
    (100, 1, 1),
    (100, 99, 50),
    (100, 98, 33),
    (100, 97, 25),
])
def test_get_goal_min_size(total_token_count, max_chunk_size, expected):
    assert CodeChunker.get_goal_min_size(total_token_count, max_chunk_size) == expected


def test_todo():
    assert False, 'Remove NODE_TYPES_TO_CHUNK and commented out references to it if I don\'t end up using it'
