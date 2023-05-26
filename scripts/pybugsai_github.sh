#!/usr/bin/env bash

set -ex
set -o pipefail

git ls-files -- '*.py' |
  grep -v ^tests/resources |
  xargs -t git diff origin/master -- |
  python -m py_bugs_open_ai.cli --diff-in --cache .pybugsai/cache
