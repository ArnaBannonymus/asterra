# Contributing

Thanks for considering contributing to Asterra.

## Development setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e ".[dev]"
```

## Running tests

```bash
pytest
```

## Linting (optional)

```bash
ruff check .
```

## Design notes

Please read `DESIGN_BOUNDARIES.md` before proposing large API additions. The library is deliberately structured to
keep EO-specific code separate from potentially generic sparse support machinery.

## Reporting issues

When reporting bugs, include:
- minimal reproducible code
- Asterra version and Python version
- expected vs actual behavior

