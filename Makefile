# XRAP developer tasks.
#
#   make help          list targets
#   make env           create the conda env from environment.yml
#   make env_export    dump the active conda env to a reproducible lock file
#   make install       editable install with dev extras (into the active env)
#   make test          run the test suite (coverage config lives in pyproject.toml)
#   make coverage      test + write an HTML coverage report to htmlcov/
#   make lint          ruff static checks
#   make format        ruff autofix + import sort
#   make build         build the sdist + wheel into dist/
#   make clean         remove build/test/cache artifacts
#
# Most targets assume the `xrap` environment is already active
# (`conda activate xrap`); `make env` is the one exception.

PYTHON       ?= python
PIP          ?= $(PYTHON) -m pip
CONDA        ?= conda
ENV_NAME     ?= xrap
ENV_LOCK     ?= environment.lock.yml
PKG          := xrap
SRC          := src/$(PKG)
TESTS        := tests

.DEFAULT_GOAL := help
.PHONY: help env env_export install test coverage lint format build clean distclean

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

env:  ## Create the conda env from environment.yml
	$(CONDA) env create -f environment.yml

env_export:  ## Dump the active conda env to $(ENV_LOCK) (full, reproducible; keeps pip deps)
	$(CONDA) env export --no-builds | grep -Ev '^prefix:|- xrap==' > $(ENV_LOCK)
	@echo "wrote $(ENV_LOCK) -- environment.yml stays the curated, hand-maintained spec"

install:  ## Editable install with dev extras into the active env
	$(PIP) install -e ".[dev]"

test:  ## Run the test suite (pytest, with coverage per pyproject.toml)
	$(PYTHON) -m pytest

coverage:  ## Run tests and write an HTML coverage report to htmlcov/
	$(PYTHON) -m pytest --cov-report=html --cov-report=term-missing
	@echo "open htmlcov/index.html"

lint:  ## Static checks (ruff)
	$(PYTHON) -m ruff check $(SRC) $(TESTS)

format:  ## Autofix lint issues and sort imports (ruff)
	$(PYTHON) -m ruff check --fix $(SRC) $(TESTS)
	$(PYTHON) -m ruff format $(SRC) $(TESTS)

build: clean  ## Build sdist + wheel into dist/
	$(PYTHON) -m build

clean:  ## Remove build/test/cache artifacts
	rm -rf build/ dist/ *.egg-info src/*.egg-info
	rm -rf .pytest_cache/ .ruff_cache/ htmlcov/
	rm -f .coverage .coverage.* coverage.xml .coverage.xml
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	find . -type f -name '*.py[co]' -delete

distclean: clean  ## clean + drop the generated version file
	rm -f $(SRC)/_version.py
