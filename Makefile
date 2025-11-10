.PHONY: format lint test

format:
	python -m black .

lint:
	ruff check .

test:
	pytest -q
