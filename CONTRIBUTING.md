# Contributing

Thanks for taking a look at this project. This repository is a Python EEG/BCI project built around MNE, scikit-learn, and custom signal-processing components.

## Setup

Install the local environment from the repository root:

```bash
make install
```

This uses `uv` through the project `Makefile` and creates local generated directories such as `.venv/`, `bin/`, `uv_python/`, and `uv_cache/`.

## Development Workflow

Run the main checks before opening a pull request:

```bash
make lint
make check
```

Format Python files with:

```bash
make format
```

Use `make clean` to remove Python caches and saved model files. Use `make fclean` only when you want to remove the local virtual environment, local `uv` install, and generated caches.

## Code Style

- Follow standard Python naming conventions.
- Keep functions and classes focused on one responsibility.
- Prefer scikit-learn-compatible APIs for pipeline components (`BaseEstimator`, `TransformerMixin`, `ClassifierMixin`) when adding new preprocessing, feature extraction, or classifier modules.
- Keep dataset-specific logic in `src/bci/datasets/` or a focused loader module.
- Do not commit generated model files, downloaded datasets, local caches, or virtual environments.

## Testing and Evaluation

For a quick subject-level check:

```bash
make train SUBJECT=1 RUNS="4 8 12"
```

For the default full evaluation:

```bash
make run
```

For BCI Competition IV Dataset 2a, make sure the `.gdf` files are available under `mne_data/BCICIV_2a_gdf/` before running:

```bash
make train DATASET=bcic4_2a SUBJECT=1 CLASSES="left_hand right_hand"
```

## Branch Naming

Use short, descriptive branch names:

```text
feat/wavelet-features
fix/epoch-extraction
docs/update-readme
```

## Commit Messages

Use concise Conventional Commit-style messages:

```text
feat(csp): implement custom spatial filters
fix(epochs): handle runs without T1/T2 events
docs(readme): clarify dataset setup
```

Common types:

- `feat`: user-facing feature or capability
- `fix`: bug fix
- `docs`: documentation-only change
- `refactor`: internal change without behavior change
- `test`: tests or evaluation scripts
- `chore`: tooling, cleanup, or maintenance

## Pull Request Checklist

Before opening a pull request:

- The README and CLI help still match the implemented behavior.
- `make lint` passes.
- `make check` passes, or any failure is explained in the PR.
- New generated files are not committed accidentally.
- Dataset licensing and citation requirements are respected.
