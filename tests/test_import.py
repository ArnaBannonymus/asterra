from __future__ import annotations


def test_import_package() -> None:
    import asterra

    from pathlib import Path
    import tomllib

    version = tomllib.loads(Path("pyproject.toml").read_text())["project"]["version"]
    assert asterra.__version__ == version
