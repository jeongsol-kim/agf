#!/usr/bin/env bash
# pi0.5 RoboCasa environment. robocasa's setup.py pins lerobot==0.3.3, so robocasa and the
# robosuite fork are installed without dependency resolution after everything else.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
pip install -r "$HERE/requirements-robocasa.txt" --extra-index-url https://download.pytorch.org/whl/cu126
pip install --no-deps \
  "robosuite @ git+https://github.com/ARISE-Initiative/robosuite.git@5ce6643f3092639d08f7b0f90ed1c6a84f50552c" \
  "robocasa @ git+https://github.com/robocasa/robocasa.git@b4684e6ee37d377cc392e98302a6b916d588b415"
