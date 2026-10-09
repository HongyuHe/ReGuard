# ReGuard

## Getting Started Instructions

### 1. CloudLab node

Please reserve an x86-64 bare-metal node at any CloudLab site with **Ubuntu 20.04**, at least 16 CPU cores, 64 GB RAM, and 100 GB storage. Please allow several days and run the sections in order. The setup requires `sudo` and a reboot; please do not run it on macOS or a personal workstation.

```bash
ssh YOUR_USERNAME@YOUR_CLOUDLAB_HOST
```

### 2. Clone the repositories

```bash
set -e
cd ~
git clone https://github.com/HongyuHe/NetNomos.git
git clone https://github.com/HongyuHe/ReGuard.git
```

### 3. Sage setup and reboot

```bash
set -e
cd ~/ReGuard/sage
bash setup_pre_reboot.sh --skip-python
sudo reboot
```

```bash
# Please wait for the node to reboot, then reconnect.
ssh YOUR_USERNAME@YOUR_CLOUDLAB_HOST
```

```bash
set -e
cd ~/ReGuard/sage
[[ "$(uname -r)" = 4.19.112-0062* ]]
bash setup_post_reboot.sh
bash setup_python_envs.sh --skip-system-packages --skip-runtime-venv
bash setup_python_envs.sh --verify-only
test -x /usr/local/bin/mm-adv-net
test -x "$HOME/venvpy36/bin/python"
test "$(sysctl -n net.ipv4.ip_forward)" = "1"
```

### 4. Park setup

```bash
set -e
export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"
pyenv install -s 3.10.14

cd ~/ReGuard/park
"$PYENV_ROOT/versions/3.10.14/bin/python" -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cpu
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -c 'import gymnasium, pandas, sklearn, torch; print("Park setup verified")'
```

### 5. Pensieve setup

```bash
set -e
export PYENV_ROOT="$HOME/.pyenv"
cd ~/ReGuard/pensieve
"$PYENV_ROOT/versions/3.10.14/bin/python" -m venv venv
venv/bin/python -m pip install --upgrade pip
venv/bin/python -m pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cpu
venv/bin/python -m pip install -r Pensieve-PPO/src/requirements.txt
venv/bin/python -m pip install pandas scikit-learn h2o z3-solver seaborn
venv/bin/python -c 'import pandas, sklearn, torch; print("Pensieve setup verified")'
test -s Pensieve-PPO/src/ppo/nn_model_ep_426300.pth
```

## Detailed Instructions

### 1. Start a persistent session

```bash
cd ~/ReGuard
tmux new -s reguard
```

### 2. Park experiment

```bash
# Please run all remaining sections in the same tmux shell.
set -e
cd ~/ReGuard/park
PARK_RUN="park_lb_seed42_ae_$(date -u +%Y%m%dT%H%M%SZ)"
PARK_RESULTS_DIR="$PWD/results/$PARK_RUN"
RUN_NAME="$PARK_RUN" bash scripts/run_reguard_experiment.sh
test -s "$PARK_RESULTS_DIR/controller_training/last.pt"
test -s "$PARK_RESULTS_DIR/shield/rules/park_shield_sklearn_risk.pl"
test -s "$PARK_RESULTS_DIR/evaluation/episode_metrics.csv"
test -s "$PARK_RESULTS_DIR/evaluation/manifest.json"
echo "Park artifacts verified: $PARK_RESULTS_DIR"
```

### 3. Pensieve experiment

```bash
set -e
cd ~/ReGuard/pensieve
PENSIEVE_RUN="pensieve_abr_seed42_ae_$(date -u +%Y%m%dT%H%M%SZ)"
PENSIEVE_RESULTS_DIR="$PWD/results/$PENSIEVE_RUN"
RUN_NAME="$PENSIEVE_RUN" bash scripts/run_reguard_experiment.sh
find "$PENSIEVE_RESULTS_DIR/controller_training" -maxdepth 1 -type f -name '*.pt' -size +0c -print -quit | grep -q .
test -s "$PENSIEVE_RESULTS_DIR/shield/rules/pensieve_bounds_sklearn_min.pl"
test -s "$PENSIEVE_RESULTS_DIR/shield/rules/pensieve_bounds_sklearn_max.pl"
test -s "$PENSIEVE_RESULTS_DIR/evaluation/manifest.json"
echo "Pensieve artifacts verified: $PENSIEVE_RESULTS_DIR"
```

### 4. Sage experiment

```bash
set -e
cd ~/ReGuard/sage
[[ "$(uname -r)" = 4.19.112-0062* ]]
SAGE_RUN="sage_cc_seed42_ae_$(date -u +%Y%m%dT%H%M%SZ)"
SAGE_RESULTS_DIR="$PWD/results/$SAGE_RUN"
RUN_NAME="$SAGE_RUN" bash scripts/run_reguard_experiment.sh
test -s "$SAGE_RESULTS_DIR/shield/dataset/sage_shield_dataset.csv"
test -s "$SAGE_RESULTS_DIR/shield/rules/sage_unified_shield_rules.json"
find "$SAGE_RESULTS_DIR/evaluation" -type f -name eval_summary.csv -size +0c -print -quit | grep -q .
test -s "$SAGE_RESULTS_DIR/evaluation/manifest.json"
echo "Sage artifacts verified: $SAGE_RESULTS_DIR"
```

## Plotting Figures 4–10

### 1. Reproduce all figures

```bash
# Please finish all three experiments before plotting.
set -e
cd ~/ReGuard
pensieve/venv/bin/python pensieve/scripts/reproduce_figures.py --results-dir "$PENSIEVE_RESULTS_DIR"
pensieve/venv/bin/python sage/scripts/reproduce_figures.py --results-dir "$SAGE_RESULTS_DIR"
pensieve/venv/bin/python park/scripts/reproduce_figures.py --results-dir "$PARK_RESULTS_DIR"
```

### 2. Verify all 17 figure pairs

```bash
set -e
cd ~/ReGuard
FIGURE_IDS=(figure_{4..8}{a,b,c} figure_9 figure_10)
for figure_id in "${FIGURE_IDS[@]}"; do
  test -s "figures/$figure_id/${figure_id}_expected.png"
  test -s "figures/$figure_id/$figure_id.pdf"
done
git diff --exit-code -- 'figures/**/*_expected.png'
echo "All 17 figure pairs are present"

command -v pdftoppm >/dev/null || sudo apt-get install -y poppler-utils
VERIFY_DIR="$(mktemp -d /tmp/reguard-figure-verification.XXXXXX)"
for figure_id in "${FIGURE_IDS[@]}"; do
  pdftoppm -png -singlefile -r 150 \
    "figures/$figure_id/$figure_id.pdf" "$VERIFY_DIR/$figure_id-reproduced"
done
echo "Rendered figures: $VERIFY_DIR"
# Please compare every rendered PNG with figures/<figure_id>/<figure_id>_expected.png.
# Please check methods, metrics, labels, legends, ordering, trends, and readability.
```

### Optional: moved results or a new shell

```bash
# Please replace each path with the current location of its complete run directory.
# --results-dir accepts an absolute path or a path relative to the current directory.
# Please pass the run directory itself, not its evaluation/ subdirectory.
# Please preserve all contents when moving a run directory.
PARK_RESULTS_DIR="/absolute/path/to/completed/park-run"
PENSIEVE_RESULTS_DIR="/absolute/path/to/completed/pensieve-run"
SAGE_RESULTS_DIR="/absolute/path/to/completed/sage-run"
# Please rerun the plotting commands above with these values.
```

### Optional: individual figures

```bash
cd ~/ReGuard
pensieve/venv/bin/python pensieve/scripts/reproduce_figures.py --results-dir "$PENSIEVE_RESULTS_DIR" --figure figure_7a
pensieve/venv/bin/python sage/scripts/reproduce_figures.py --results-dir "$SAGE_RESULTS_DIR" --figure figure_7b
pensieve/venv/bin/python park/scripts/reproduce_figures.py --results-dir "$PARK_RESULTS_DIR" --figure figure_7c
```

### Optional: reconnect to an active experiment

```bash
ssh YOUR_USERNAME@YOUR_CLOUDLAB_HOST
```

```bash
tmux attach -t reguard
```
