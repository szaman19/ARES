#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKTREE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Reuse main repo .venv when running inside a git worktree without a local .venv
if [[ -x "${WORKTREE_ROOT}/.venv/bin/python" ]]; then
  PY_BIN="${WORKTREE_ROOT}/.venv/bin/python"
elif [[ -x "${WORKTREE_ROOT}/../../.venv/bin/python" ]]; then
  PY_BIN="${WORKTREE_ROOT}/../../.venv/bin/python"
else
  PY_BIN="python3"
fi

cd "${WORKTREE_ROOT}"

if [[ ! -d "${WORKTREE_ROOT}/.data_cache/btsbot/observations" ]]; then
  echo "[setup] Staging safe multimodal dataset slices into ./.data_cache ..."
  "${PY_BIN}" scripts/stage_multimodal_datasets.py --dataset all --cache-dir ./.data_cache --media-limit 48
fi

exec "${PY_BIN}" -m multimodal_demo.app "$@"
