"""The 2-minute judge demo must keep working on the committed sample."""
from shift import quickstart
from shift.config import REPO_ROOT


def test_quickstart_runs_on_sample(capsys):
    assert (REPO_ROOT / "samples" / "florida_sample.zip").exists()
    quickstart.main()
    out = capsys.readouterr().out
    assert "[calibrate] VIIRS_FRP =" in out
    assert "region_month" in out
