#!/bin/sh
set -euo pipefail

pytest -q
ruff check .
mypy --ignore-missing-imports
