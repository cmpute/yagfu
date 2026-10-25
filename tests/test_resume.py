"""Tests for yagfu.log tee'ing, band-file appending, and NFS checkpoint
state — no GPU needed."""

from yagfu.config import RunContext


# ------------------------------------------------------------- yagfu.log

def test_log_tees_to_file(tmp_path):
    lf = tmp_path / "yagfu.log"
    ctx = RunContext(log_file=lf)
    ctx.log("first")
    ctx.logv("detail")            # verbose=0: not logged anywhere
    assert lf.read_text().endswith("first\n")
    assert "detail" not in lf.read_text()
    ctx.verbose = 1
    ctx.logv("detail2")
    assert "detail2" in lf.read_text()


def test_log_appends_across_contexts(tmp_path):
    lf = tmp_path / "yagfu.log"
    RunContext(log_file=lf).log("one")
    RunContext(log_file=lf).log("two")
    text = lf.read_text()
    assert "one" in text and "two" in text


def test_quiet_still_logs_to_file(tmp_path):
    lf = tmp_path / "yagfu.log"
    ctx = RunContext(quiet=True, log_file=lf)
    ctx.log("recorded")
    assert "recorded" in lf.read_text()   # the file is the record


def test_log_file_failure_is_silent(tmp_path):
    ctx = RunContext(log_file=tmp_path / "no" / "such" / "dir" / "y.log")
    ctx.log("still works")                 # must not raise


# -------------------------------------------------------- band appending

def test_append_band_rels(tmp_path):
    from yagfu.backends.nfs import _append_band_rels
    a = tmp_path / "band1.0.rels"; a.write_bytes(b"r1\n")
    b = tmp_path / "band1.1.rels"; b.write_bytes(b"r2\nr3\n")
    _append_band_rels(tmp_path, ["band1.0.rels", "band1.1.rels"])
    dat = tmp_path / "msieve.dat"
    assert dat.read_bytes() == b"r1\nr2\nr3\n"
    assert not a.exists() and not b.exists()      # bands deleted


def test_append_band_rels_appends_existing(tmp_path):
    from yagfu.backends.nfs import _append_band_rels
    (tmp_path / "msieve.dat").write_bytes(b"old\n")
    a = tmp_path / "band2.0.rels"; a.write_bytes(b"new\n")
    _append_band_rels(tmp_path, ["band2.0.rels"])
    assert (tmp_path / "msieve.dat").read_bytes() == b"old\nnew\n"


# ------------------------------------------------------------- nfs state

def test_nfs_state_roundtrip(tmp_path):
    from yagfu.backends.nfs import _load_nfs_state, _save_nfs_state
    _save_nfs_state(tmp_path, 12345, q_next=99, target_raw=1000,
                    have_raw=500, round_idx=3, bands=4, last_cnt=250,
                    last_width=30)
    s = _load_nfs_state(tmp_path, 12345)
    assert s["q_next"] == 99 and s["target_raw"] == 1000
    assert s["have_raw"] == 500 and s["bands"] == 4
    assert _load_nfs_state(tmp_path, 99999) is None   # other number


def test_nfs_state_corrupt(tmp_path):
    from yagfu.backends.nfs import _load_nfs_state
    (tmp_path / "nfs-state.json").write_text("{not json")
    assert _load_nfs_state(tmp_path, 1) is None


def test_nfs_resume_pending(tmp_path):
    from yagfu.backends.nfs import nfs_resume_pending
    assert nfs_resume_pending(tmp_path, 12345) is False
    job = tmp_path / "c5.job"
    job.write_text("n: 67890\nrlim: 1\n")
    assert nfs_resume_pending(tmp_path, 12345) is False   # different n
    job.write_text("n: 12345\nrlim: 1\n")
    assert nfs_resume_pending(tmp_path, 12345) is True
