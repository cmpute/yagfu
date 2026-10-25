import pytest

from yagfu.cli import build_parser, load_numbers
from yagfu.config import RunContext


def test_ecm_only_flag():
    ap = build_parser()
    args = ap.parse_args(["--ecm-only", "12345"])
    assert args.ecm_only is True
    args = ap.parse_args(["12345"])
    assert args.ecm_only is False


def test_ctx_defaults():
    ctx = RunContext()
    assert ctx.ecm_only is False
    assert ctx.quiet is False
    assert ctx.verbose == 0


def test_log_levels(capsys):
    ctx = RunContext()
    ctx.log("key")                 # default: shown
    ctx.logv("detail")             # needs -v: hidden
    err = capsys.readouterr().err
    assert "key" in err and "detail" not in err
    ctx.verbose = 1
    ctx.logv("detail2")
    assert "detail2" in capsys.readouterr().err
    quiet = RunContext(quiet=True)
    quiet.log("hidden")
    assert capsys.readouterr().err == ""


def test_load_numbers_file(tmp_path):
    f = tmp_path / "nums.txt"
    f.write_text("# comment\n\n123456789\n  98765 \n")
    args = build_parser().parse_args(["-f", str(f)])
    assert load_numbers(args) == [123456789, 98765]


def test_load_numbers_mutually_exclusive(tmp_path, capsys):
    f = tmp_path / "n.txt"
    f.write_text("6\n")
    args = build_parser().parse_args(["-f", str(f), "10"])
    with pytest.raises(SystemExit):
        load_numbers(args)
