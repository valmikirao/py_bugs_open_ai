#!/usr/bin/env bash

set -ex
set -o pipefail

PY_DIR=.venv

if [[ ! -d .venv ]]; then
  python -m venv .venv
fi
PY_CMD="${PY_DIR}/bin/python"

"$PY_CMD" -m pip install  'pip==23.1.2'
"$PY_CMD" -m pip install .
git ls-files -- '*.py' |
  grep -v ^tests/resources |
  xargs -t git diff master -- |
  "$PY_CMD" -m py_bugs_open_ai.cli --diff-in --cache .pybugsai/cache
