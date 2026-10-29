.PHONY: check format lint typecheck

# 린트 + 포맷 검사 + 타입 검사 (파일은 수정하지 않음)
check: lint typecheck

lint:
	uv run ruff check .
	uv run ruff format --check .

typecheck:
	uv run mypy .

# import 정렬 등 자동 수정 가능한 린트 오류를 고친 뒤 포맷
format:
	uv run ruff check --fix .
	uv run ruff format .
