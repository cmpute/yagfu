import json

import pytest

from yagfu.fdb import FactorDbClient, FactorDbError, read_token


class FakeTransport:
    """Records rpc calls; returns canned results per method."""

    def __init__(self, results=None, errors=None):
        self.calls = []
        self.results = results or {}
        self.errors = errors or {}

    def __call__(self, client, method, params, timeout=None):
        self.calls.append((method, params))
        if method in self.errors:
            raise FactorDbError(self.errors[method])
        return self.results.get(method)


@pytest.fixture
def client(monkeypatch):
    c = FactorDbClient(token=None, delay=0.0)
    transport = FakeTransport()
    monkeypatch.setattr(c, "rpc", lambda method, params, timeout=None:
                        transport(c, method, params, timeout))
    c._transport = transport
    return c


def test_get_number_parses_record(client):
    client._transport.results["get_number"] = {
        "id": {"fid": "abc123"}, "status": "C",
        "decimal": "123456789", "digits": 9}
    rec = client.get_number(123456789)
    assert rec == {"fid": "abc123", "status": "C",
                   "decimal": "123456789", "digits": 9, "snfs": None}


def test_get_number_by_id(client):
    client._transport.results["get_number"] = {
        "id": {"fid": 1100000000631376229}, "status": "C",
        "decimal": "1166367688", "digits": 135}
    rec = client.get_number_by_id("1100000000631376229")
    assert rec == {"fid": 1100000000631376229, "status": "C",
                   "decimal": "1166367688", "digits": 135, "snfs": None}
    (method, params), = client._transport.calls
    assert method == "get_number"
    assert params["target"] == {"id": 1100000000631376229}
    assert params["decimal"] is True


def test_get_number_by_id_rejects_garbage(client):
    assert client.get_number_by_id("12x3") is None
    assert client.get_number_by_id(None) is None
    assert client._transport.calls == []


def test_get_number_by_id_without_decimal_ignored(client):
    # a record with an omitted decimal carries no number to factor
    client._transport.results["get_number"] = {
        "id": {"fid": 7}, "status": "C", "digits": 9,
        "decimal_omitted": True}
    assert client.get_number_by_id(7) is None


def test_get_number_snfs_request_and_passthrough(client):
    snfs = {"n": "6", "polys": [{"c": ["1", "1"]}], "status": "C"}
    client._transport.results["get_number"] = {
        "id": {"fid": "f"}, "status": "C", "decimal": "6", "digits": 1,
        "snfs": snfs}
    rec = client.get_number(6, snfs=True)
    (_, params), = client._transport.calls
    assert params["snfs"] is True
    assert params["detail"] == 2
    assert rec["snfs"] is snfs
    # the plain query asks for neither key; a stray field still passes
    # through (using it is the caller's policy)
    client._transport.calls.clear()
    client._transport.results["get_number"] = {
        "id": {"fid": "f"}, "status": "C", "decimal": "6", "digits": 1,
        "snfs": snfs}
    rec = client.get_number(6)
    (_, params), = client._transport.calls
    assert "snfs" not in params and "detail" not in params
    assert rec["snfs"] is snfs


def test_get_number_mismatch_ignored(client):
    client._transport.results["get_number"] = {
        "id": {"fid": "x"}, "status": "C", "decimal": "999", "digits": 3}
    assert client.get_number(123456789) is None


def test_get_factor_decimals(client):
    client._transport.results["get_factors"] = {
        "factors": [{"base": "2", "fid": "f2"},
                    {"base": "1234567890123456789012345678901234567891", "fid": "f3"}]}
    assert client.get_factor_decimals("p") == ["2", "1234567890123456789012345678901234567891"]


def test_report_factors_payload_no_token(client):
    client._transport.results["report_factors"] = {"status": "C", "credited": False}
    res = client.report_factors("fid1", 123, [3, 41])
    (method, params), = client._transport.calls
    assert method == "report_factors"
    assert params == {"target": {"id": "fid1"}, "factors": ["3", "41"]}
    assert "credit" not in params


def test_report_factors_payload_with_token(monkeypatch):
    c = FactorDbClient(token="tok", delay=0.0)
    transport = FakeTransport({"report_factors": {"status": "FF"}})
    monkeypatch.setattr(c, "rpc", lambda method, params, timeout=None:
                        transport(c, method, params, timeout))
    c.report_factors(None, 123, [3])
    (method, params), = transport.calls
    assert params["target"] == {"expr": "123"}
    assert params["credit"] is True


def test_report_ecm_payload(client):
    client._transport.results["report_ecm"] = {"ok": True}
    client.report_ecm("ffid", b1=11000, sigma=12345, b2=1873422)
    (method, params), = client._transport.calls
    assert method == "report_ecm"
    assert params == {"id": "ffid", "method": "ecm", "param": 3,
                      "b1": "11000", "sigma": "12345", "b2": "1873422"}


def test_report_ecm_stage1_no_b2(client):
    client._transport.results["report_ecm"] = {"ok": True}
    client.report_ecm("ffid", b1=2000, sigma=7, b2=None)
    (_, params), = client._transport.calls
    assert "b2" not in params


def test_dry_run_sends_nothing():
    c = FactorDbClient(dry_run=True, delay=0.0)
    assert c.rpc("get_number", {"target": {"expr": "6"}}) is None


def test_error_wrapped(client):
    client._transport.errors["get_number"] = FactorDbError("boom")
    assert client.get_number(6) is None  # query errors degrade to None
    client._transport.errors["report_factors"] = FactorDbError("boom")
    # report errors are also non-fatal: logged, return None
    assert client.report_factors("f", 6, [2, 3]) is None


def test_read_token(tmp_path, monkeypatch):
    p = tmp_path / "tok"
    p.write_text("  abc\n")
    assert read_token(p) == "abc"
    p.write_text("\n")
    assert read_token(p) is None
    assert read_token(tmp_path / "missing") is None
    monkeypatch.setenv("HOME", str(tmp_path))
    assert read_token(None) is None


def test_rpc_envelope_and_headers(monkeypatch):
    captured = {}

    class FakeUrlopen:
        def __init__(self, req, timeout=None):
            captured["url"] = req.full_url
            captured["data"] = json.loads(req.data.decode())
            captured["headers"] = req.headers
            self._body = json.dumps({"jsonrpc": "2.0", "id": 1,
                                     "result": {"ok": 1}}).encode()

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return self._body

    monkeypatch.setattr("urllib.request.urlopen", FakeUrlopen)
    c = FactorDbClient(token="T", delay=0.0)
    res = c.rpc("report_ecm", {"id": "x"})
    assert res == {"ok": 1}
    assert captured["url"] == "https://factordb.com:4059/rpc"
    assert captured["data"]["method"] == "report_ecm"
    assert captured["data"]["jsonrpc"] == "2.0"
    assert captured["headers"]["X-fdb-user-token"] == "T"
    assert captured["headers"]["User-agent"].startswith("yagfu/")
