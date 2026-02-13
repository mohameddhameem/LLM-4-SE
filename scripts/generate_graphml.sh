#!/bin/bash
export PYTHONIOENCODING=utf-8
export PYTHONUTF8=1

export PYTHONPATH="$PWD:$PYTHONPATH"

# Default to 4 workers, can be overridden: WORKERS=8 bash scripts/generate_graphml.sh
WORKERS=${WORKERS:-128}

python Joern.py --workers $WORKERS