#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python3 "$ROOT/code/run_pooled_delegation_model.py"
python3 "$ROOT/code/summarise_paper_figures.py"
