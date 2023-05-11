#!/usr/bin/env bash

set -ex

PY_DIR=.venv

if [[ ! -d .venv ]]; then
  python -m venv .venv
fi
PY_CMD="${PY_DIR}/bin/python"

"$PY_CMD" -m pip install  'pip==23.1.2'
"$PY_CMD" -m pip install .
git ls-files -- '*.py' | "$PY_CMD" -m py_bugs_open_ai.cli --in --cache .pybugsai/cache