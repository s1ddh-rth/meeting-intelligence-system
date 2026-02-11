.PHONY: run stop test lint clean dev

run:
	docker-compose up --build

run-detached:
	docker-compose up --build -d

stop:
	docker-compose down

test:
	pytest tests/ -v

lint:
	ruff check src/ tests/

clean:
	docker-compose down -v
	rm -rf data/sqlite/*.db

dev:
	uvicorn src.app:app --host 0.0.0.0 --port 8000 --reload
