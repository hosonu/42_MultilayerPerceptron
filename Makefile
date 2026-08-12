# ==============================================================================
# 42 Multilayer Perceptron - Makefile
# ==============================================================================

# Variables
LOCAL_BIN   := ./bin
VENV        := ./.venv
UV          := $(LOCAL_BIN)/uv
UV_PY       := ./uv_python
UV_CACHE    := ./uv_cache
PYTHON      := $(UV) run python
FLAKE8      := $(UV) run flake8
AUTOPEP8    := $(UV) run autopep8
PYTEST      := $(UV) run pytest

# Virtual environment
export UV_PYTHON_INSTALL_DIR := $(UV_PY)
export UV_CACHE_DIR := $(UV_CACHE)
export UV_PYTHON_DOWNLOADS := auto

# Directories
SRC_DIR     := src
SCRIPTS_DIR := scripts
TESTS_DIR   := tests

# Entry points
SPLIT_PY    := split.py
TRAIN_PY    := train.py
PREDICT_PY  := predict.py
EXPLORE_PY  := $(SCRIPTS_DIR)/explore_dataset.py
SWEEP_PY    := $(SCRIPTS_DIR)/sweep_architectures.py
SWEEP_SEEDS_PY := $(SCRIPTS_DIR)/sweep_seeds.py
SWEEP_KFOLD_PY := $(SCRIPTS_DIR)/sweep_kfold.py

# Optional CLI passthrough: make train ARGS="--epochs 500"
ARGS        ?=

# Rules
.PHONY: all help install pre sync split train predict explore sweep sweep-seeds sweep-kfold run \
        lint format fmt check test clean fclean re

all: install

help:
	@echo "42 Multilayer Perceptron — Commands"
	@echo ""
	@echo "  Setup:"
	@echo "    make install        Install dependencies using uv"
	@echo ""
	@echo "  Pipeline:"
	@echo "    make split          Split data/data.csv into train/validation CSVs"
	@echo "    make train          Train the MLP and save model.pkl"
	@echo "    make predict        Evaluate a saved model"
	@echo "    make explore        Explore the dataset (EDA plots)"
	@echo "    make sweep          Grid-search architectures / hyperparameters"
	@echo "    make sweep-seeds    Fixed hyperparams, sweep seeds only"
	@echo "    make sweep-kfold    K-Fold CV sweep for robust hyperparameter search"
	@echo "    make run            Alias for make train"
	@echo ""
	@echo "  Pass flags with ARGS, e.g.:"
	@echo "    make train ARGS=\"--epochs 500 --no-plot\""
	@echo "    make sweep ARGS=\"--max-trials 4 --lrs 0.01 --batch-sizes 32 --units 16 24\""
	@echo "    make sweep-seeds ARGS=\"--seeds 42 0 1 7 123\""
	@echo ""
	@echo "  Development:"
	@echo "    make lint           Run flake8 on $(SRC_DIR), $(SCRIPTS_DIR), and $(TESTS_DIR)"
	@echo "    make format / fmt   Run autopep8 formatter"
	@echo "    make test           Run pytest"
	@echo "    make check          Run lint and tests"
	@echo ""
	@echo "  Cleanup:"
	@echo "    make clean          Remove Python caches and model.pkl"
	@echo "    make fclean         Remove .venv, local uv, and generated caches"
	@echo "    make re             Full reinstallation"

$(UV):
	@echo "Downloading uv locally into ./bin..."
	@mkdir -p $(LOCAL_BIN)
	curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="$(abspath $(LOCAL_BIN))" INSTALLER_NO_MODIFY_PATH=1 UV_NO_MODIFY_PATH=1 sh

# Dependency Management
pre:
	@echo "Creating directories in this project directory..."
	@mkdir -p $(UV_PY)
	@mkdir -p $(UV_CACHE)

sync: pre $(UV)
	@echo "Syncing environment with uv.lock..."
	$(UV) sync

install: sync
	@echo "Done! Environment is ready."

# Pipeline
split: sync
	@echo "==> Splitting dataset..."
	$(PYTHON) $(SPLIT_PY) $(ARGS)

train: sync
	@echo "==> Training MLP..."
	$(PYTHON) $(TRAIN_PY) $(ARGS)

predict: sync
	@echo "==> Running prediction..."
	$(PYTHON) $(PREDICT_PY) $(ARGS)

explore: sync
	@echo "==> Exploring dataset..."
	$(PYTHON) $(EXPLORE_PY) $(ARGS)

sweep: sync
	@echo "==> Sweeping architectures..."
	$(PYTHON) $(SWEEP_PY) $(ARGS)

sweep-seeds: sync
	@echo "==> Sweeping seeds for fixed hyperparameters..."
	$(PYTHON) $(SWEEP_SEEDS_PY) $(ARGS)

sweep-kfold: sync
	@echo "==> K-Fold CV sweep (robust hyperparameter search)..."
	$(PYTHON) $(SWEEP_KFOLD_PY) $(ARGS)

run: train

# Development Tools
lint: sync
	$(FLAKE8) $(SRC_DIR) $(SCRIPTS_DIR) $(TESTS_DIR)

format fmt: sync
	$(AUTOPEP8) -i -r $(SRC_DIR) $(SCRIPTS_DIR) $(TESTS_DIR)

test: sync
	$(PYTEST) $(TESTS_DIR)

check: sync lint test

# Cleanup
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -f model.pkl

fclean: clean
	@echo "Cleaning up..."
	rm -rf $(VENV)
	rm -rf $(LOCAL_BIN)
	rm -rf $(UV_PY)
	rm -rf $(UV_CACHE)
	@echo "Clean complete."

re: fclean all
