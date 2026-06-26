.PHONY: install lint test e2e clean

install:
	pip install -e ".[dev]"

lint:
	ruff check src/ tests/
	ruff format --check src/ tests/

fmt:
	ruff check --fix src/ tests/
	ruff format src/ tests/

test:
	pytest tests/ -v --tb=short

e2e:
	RDF_MODELS=mock RDF_STORAGE=local RDF_QUEUE=local RDF_CATALOG=local \
	pytest tests/test_e2e.py -v --tb=short

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; true
	find . -type f -name "*.pyc" -delete 2>/dev/null; true
	rm -rf .pytest_cache dist build *.egg-info src/*.egg-info
