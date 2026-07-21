# ==============================================================================
# 42 Template Python - Makefile
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

# Entry point — update this when you add your main script
MAIN_PY     := main.py

# Rules
.PHONY: all help install pre sync run lint format fmt check test clean fclean re

all: install

help:
	@echo "42 Template Python — Commands"
	@echo ""
	@echo "  Setup:"
	@echo "    make install        Install dependencies using uv"
	@echo ""
	@echo "  Execution:"
	@echo "    make run            Run the main entry point ($(MAIN_PY))"
	@echo ""
	@echo "  Development:"
	@echo "    make lint           Run flake8 on $(SRC_DIR), $(SCRIPTS_DIR), and $(TESTS_DIR)"
	@echo "    make format / fmt   Run autopep8 formatter"
	@echo "    make test           Run pytest"
	@echo "    make check          Run lint and tests"
	@echo ""
	@echo "  Cleanup:"
	@echo "    make clean          Remove Python cache files"
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

# Main Tasks
run: sync
	@echo "==> Running $(MAIN_PY)..."
	$(PYTHON) $(MAIN_PY)

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

fclean: clean
	@echo "Cleaning up..."
	rm -rf $(VENV)
	rm -rf $(LOCAL_BIN)
	rm -rf $(UV_PY)
	rm -rf $(UV_CACHE)
	@echo "Clean complete."

re: fclean all
