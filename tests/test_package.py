"""Smoke tests for the package skeleton."""

from typer.testing import CliRunner

import openmind_plug
from openmind_plug.cli import app

runner = CliRunner()


def test_version_is_exposed() -> None:
    assert isinstance(openmind_plug.__version__, str)
    assert openmind_plug.__version__


def test_cli_version() -> None:
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert openmind_plug.__version__ in result.output


def test_cli_help() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "list-sources" in result.output
