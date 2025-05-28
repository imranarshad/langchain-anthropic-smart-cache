.PHONY: help install install-dev test lint format clean build upload upload-test

help:
	@echo "Available commands:"
	@echo "  install      Install package dependencies"
	@echo "  install-dev  Install development dependencies"
	@echo "  test         Run tests"
	@echo "  lint         Run linting"
	@echo "  format       Format code with black"
	@echo "  clean        Clean build artifacts"
	@echo "  build        Build package"
	@echo "  upload-test  Upload to TestPyPI"
	@echo "  upload       Upload to PyPI"

install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements-dev.txt

test:
	pytest tests/ -v --cov=langchain_anthropic_smart_cache

lint:
	flake8 langchain_anthropic_smart_cache/
	mypy langchain_anthropic_smart_cache/

format:
	black langchain_anthropic_smart_cache/ tests/ examples/

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

build: clean
	python -m build

upload-test: build
	twine upload --repository testpypi dist/*

upload: build
	twine upload dist/*