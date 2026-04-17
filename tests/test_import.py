from __future__ import annotations


def test_import_package() -> None:
    import asterra

    assert asterra.__version__ == "0.1.0"

