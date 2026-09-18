from mcsirius.cli import main


def test_cli_runs_complete_dry_run(
    capsys,
):
    result = main(["27"])

    assert result == 0

    output = capsys.readouterr().out

    assert "MODE        : DRY RUN" in output
    assert "Ion mass    : 27 u" in output
    assert "Sputter     : 6.000 kV" in output
    assert "Extraction  : 17.000 kV" in output
    assert "Einzel      : 17.500 kV" in output
    assert "Cup 1" in output
    assert "Converged   : yes" in output


def test_explicit_dry_run_flag(
    capsys,
):
    result = main(
        [
            "40",
            "--dry-run",
        ]
    )

    assert result == 0

    output = capsys.readouterr().out

    assert "Ion mass    : 40 u" in output
    assert "MODE        : DRY RUN" in output


def test_cli_rejects_non_positive_mass(
    capsys,
):
    result = main(["0"])

    assert result == 2

    output = capsys.readouterr().out

    assert "greater than zero" in output