import needle


def test_package_exposes_version() -> None:
    assert needle.__version__ == "0.1.0"
