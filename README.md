# Official implementation of AGF

The cleanup and refactoring of this codebase were assisted by a large language model (LLM).

## Setup

Python 3.12.9, torch 2.8.0+cu126. Each requirements file is a full `pip freeze`.

### 1. Environments

Use a separate environment per row.

| Environment | Used for | Install |
| --- | --- | --- |
| lerobot | SmolVLA, MolmoAct2, Q training | `pip install -r envs/requirements-lerobot.txt` |
| pi05 | pi0.5 on LIBERO-PRO, pi0.5 RoboCasa AGF training | `pip install -r envs/requirements-pi05.txt` |
| robocasa | pi0.5 RoboCasa collection and evaluation | `bash envs/install_robocasa.sh` |

Add `--extra-index-url https://download.pytorch.org/whl/cu126` to the `pip install` commands.
Model weights and the LeRobot LIBERO dataset are downloaded from Hugging Face on first use.

### 2. LIBERO config

Run once in the lerobot or pi05 environment, and answer `N` to keep the default paths:

```bash
python -c "import libero.libero"   # creates ~/.libero/config.yaml
```

### 3. LIBERO-PRO

Clone, apply the fixes in `envs/libero_pro_fixes.patch`, and point a LIBERO config at the checkout:

```bash
export LIBERO_PRO_ROOT=/path/to/LIBERO-PRO

git clone https://github.com/Zxy-MLlab/LIBERO-PRO "$LIBERO_PRO_ROOT"
cd "$LIBERO_PRO_ROOT"
git checkout eafdb80
git apply /path/to/agf/envs/libero_pro_fixes.patch

mkdir -p .libero_runtime
cat > .libero_runtime/config.yaml <<EOF
benchmark_root: $LIBERO_PRO_ROOT/libero/libero
bddl_files: $LIBERO_PRO_ROOT/libero/libero/bddl_files
init_states: $LIBERO_PRO_ROOT/libero/libero/init_files
assets: $LIBERO_PRO_ROOT/libero/libero/assets
datasets: $LIBERO_PRO_ROOT/datasets
EOF
```

Leave `LIBERO_CONFIG_PATH` unset. Collection and MolmoAct2 evaluation use this config;
SmolVLA/pi0.5 LIBERO-PRO evaluation uses `~/.libero`, as in the original runs.

### 4. RoboCasa (pi0.5 only)

In the robocasa environment:

```bash
python -m robocasa.scripts.setup_macros
python -m robocasa.scripts.download_kitchen_assets   # large download
```

### 5. MolmoAct2 evaluation only

MolmoAct2 evaluation uses a different LeRobot checkout, placed first on the path:

```bash
git clone https://github.com/allenai/lerobot allenai-lerobot
cd allenai-lerobot
git fetch origin pull/2/head
git checkout 80633827176a0203064cb141383664fba024e050

PYTHONPATH=/path/to/allenai-lerobot/src python evaluate.py --vla molmoact2 ...
```

## Quick start

SmolVLA on `libero_goal` task 0. Run from this directory; each step uses the previous step's output.

```bash
# 1. Collect rollouts (all tasks of the suite)
python collect_rollouts.py --vla smolvla --env libero --suites libero_goal --output_root critic_data

# 2. Train Q
python train_q.py --env libero --suite libero_goal --task_id 0 \
  --critic_data_root critic_data --output_dir outputs/q/libero_goal_0

# 3. Train AGF (VLA and Q frozen)
python train_agf.py --vla smolvla --env libero --libero_suite libero_goal --task_id 0 \
  --critic_data_root critic_data \
  --critic_ckpt outputs/q/libero_goal_0/qpilots_libero_q_step010000.pt \
  --output_dir outputs/agf/libero_goal_0

# 4. Evaluate
python evaluate.py --vla smolvla --env libero_pro --base_suite libero_goal --task_ids 0 \
  --conditions clean --mode agf \
  --critic_ckpt_template 'outputs/q/{suite}_{task}/qpilots_libero_q_step010000.pt' \
  --gradnet_ckpt_template 'outputs/agf/{suite}_{task}/agf_discrete_guidance.pt' \
  --output_dir outputs/evaluation
```

`--help` on any script lists its options. Defaults follow the paper experiments.

| VLA | Collection | AGF training | Evaluation |
| --- | --- | --- | --- |
| `smolvla` | `libero`, `libero_pro` | `libero`, `libero_pro` | `libero`, `libero_pro` |
| `pi05` | `libero_pro`, `robocasa` | `libero`, `libero_pro`, `robocasa` | `libero_pro`, `robocasa` |
| `molmoact2` | `libero_pro` | `libero`, `libero_pro` | `libero`, `libero_pro` |

- Clean LIBERO for pi0.5/MolmoAct2: `--env libero_pro --conditions clean`.
- MolmoAct2 Q training: add `--image2_key observation.images.wrist_image`.
- Offline Q evaluation: `evaluate_q.py`.

## Structure

```text
collect_rollouts.py, train_q.py, train_agf.py, evaluate.py, evaluate_q.py   entry points
options.py   command-line options
critics/     Q networks and losses
guidance/    guidance networks, AGF targets, guided sampling
tasks/       datasets and environment helpers
training/    AGF data loading and logging
```
