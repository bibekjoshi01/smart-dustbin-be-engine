# Backend Engine For Smart Dustbin

## Formatting and Linting Code

   ```pre-commit install```

1. ruff check / ruff check --fix / ruff format
2. black .
3. pre-commit run --all-files

## Running Tests

   ```bash
   pytest -v -s
   ```



## Run

uvicorn app.main:app --reload
