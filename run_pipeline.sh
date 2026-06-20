#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [ -d "pr-agent-env" ]; then
  source pr-agent-env/bin/activate
fi

cd description-generation
python main.py --limit 10 --randomize

cd ../judge
python judge.py
