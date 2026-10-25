"""factordb.com client — JSON-RPC 2.0 transport.

Endpoint and payload shapes follow the reference implementation
(/media/nas/users/jacob/factor/fdb_yafu_loop.py + fetch_composites.py):

* POST https://factordb.com:4059/rpc  with {"jsonrpc":"2.0","id":1,
  "method":..., "params":...}
* optional token (contents of ~/.factordb_token) in X-Fdb-User-Token;
  when present, reports are sent with credit.
* methods: get_number, get_factors, report_factors, report_ecm
* politeness: >=0.5 s between calls, exponential backoff (5 s -> 300 s,
  8 retries) on HTTP 429/5xx/network errors, Retry-After honored.

Record status codes: C (composite, no factors known), CF (composite,
factors known), FF (fully factored), P (prime), PR (probable prime),
U (unknown).
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Callable

from . import __version__

RPC_URL = "https://factordb.com:4059/rpc"


def _default_token_path() -> Path:
    return Path.home() / ".factordb_token"

#: report_ecm accepts stored primes of 25..100 digits only
ECM_REPORT_MIN_DIGITS = 25
ECM_REPORT_MAX_DIGITS = 100


def read_token(path: Path | str | None = None) -> str | None:
    p = Path(path) if path else _default_token_path()
    try:
        tok = p.read_text().strip()
    except OSError:
        return None
    return tok or None


class FactorDbError(Exception):
    pass


class FactorDbClient:
    def __init__(self, token: str | None = None, *, dry_run: bool = False,
                 delay: float = 0.5, timeout: float = 60.0,
                 retries: int = 8, log: Callable[[str], None] | None = None):
        self.token = token
        self.dry_run = dry_run
        self.delay = delay
        self.timeout = timeout
        self.retries = retries
        self._log = log or (lambda msg: None)
        self._last_call = 0.0

    # ------------------------------------------------------------ transport

    def rpc(self, method: str, params: dict, timeout: float | None = None):
        if self.dry_run:
            self._log(f"fdb[dry-run] {method} {json.dumps(params)[:500]}")
            return None
        self._throttle()
        payload = json.dumps({"jsonrpc": "2.0", "id": 1,
                              "method": method, "params": params}).encode()
        headers = {"Content-Type": "application/json",
                   "User-Agent": f"yagfu/{__version__}"}
        if self.token:
            headers["X-Fdb-User-Token"] = self.token
        backoff = 5.0
        for attempt in range(self.retries + 1):
            req = urllib.request.Request(RPC_URL, data=payload, headers=headers)
            try:
                with urllib.request.urlopen(req, timeout=timeout or self.timeout) as resp:
                    doc = json.loads(resp.read().decode())
            except urllib.error.HTTPError as e:
                if e.code == 429 or 500 <= e.code < 600:
                    wait = backoff
                    ra = e.headers.get("Retry-After") if e.headers else None
                    if ra:
                        try:
                            wait = max(wait, float(ra))
                        except ValueError:
                            pass
                    self._log(f"fdb: HTTP {e.code}; retrying in {wait:.0f}s")
                    time.sleep(wait)
                    backoff = min(backoff * 2, 300)
                    continue
                raise FactorDbError(f"fdb HTTP {e.code}") from e
            except (urllib.error.URLError, TimeoutError, OSError) as e:
                if attempt >= self.retries:
                    raise FactorDbError(f"fdb network error: {e!r}") from e
                self._log(f"fdb: network error ({e!r}); retrying in {backoff:.0f}s")
                time.sleep(backoff)
                backoff = min(backoff * 2, 300)
                continue
            if "error" in doc:
                raise FactorDbError(f"fdb rpc error: {doc['error']}")
            self._last_call = time.monotonic()
            return doc["result"]
        raise FactorDbError("fdb: gave up after retries")

    def _throttle(self) -> None:
        wait = self.delay - (time.monotonic() - self._last_call)
        if wait > 0:
            time.sleep(wait)

    # ------------------------------------------------------------- queries

    def get_number(self, n: int) -> dict | None:
        """Record for decimal n: {fid, status, decimal, digits} or None."""
        try:
            res = self.rpc("get_number",
                           {"target": {"expr": str(n)}, "decimal": True},
                           timeout=120)
        except FactorDbError as e:
            self._log(f"fdb: get_number failed: {e}")
            return None
        if not res:
            return None
        fid = (res.get("id") or {}).get("fid")
        dec = res.get("decimal")
        if dec and str(dec) != str(n):
            self._log("fdb: decimal mismatch, ignoring record")
            return None
        return {"fid": fid, "status": res.get("status"),
                "decimal": str(dec) if dec else str(n),
                "digits": res.get("digits")}

    def get_factor_decimals(self, fid: str) -> list[str]:
        """Decimal strings of the factors stored for record `fid`."""
        try:
            res = self.rpc("get_factors", {"target": {"id": fid}})
        except FactorDbError as e:
            self._log(f"fdb: get_factors failed: {e}")
            return []
        out = []
        for x in (res or {}).get("factors") or []:
            base = x.get("base")
            if base is not None:
                out.append(str(base))
        return out

    def factor_fid_map(self, fid: str) -> dict[str, str]:
        """map factor decimal -> its own record fid (for report_ecm)."""
        try:
            res = self.rpc("get_factors", {"target": {"id": fid}})
        except FactorDbError as e:
            self._log(f"fdb: get_factors failed: {e}")
            return {}
        return {str(x.get("base")): x.get("fid")
                for x in (res or {}).get("factors") or []
                if x.get("base") is not None}

    # ------------------------------------------------------------- reports

    def report_factors(self, fid: str | None, n: int,
                       factors: list[int]) -> dict | None:
        """Report prime/PRP factors of n; credit when a token is present."""
        if not factors:
            return None
        params: dict = {"target": {"id": fid} if fid else {"expr": str(n)},
                        "factors": [str(f) for f in factors]}
        if self.token:
            params["credit"] = True
        try:
            res = self.rpc("report_factors", params)
        except FactorDbError as e:
            self._log(f"fdb: report_factors failed: {e}")
            return None
        self._log(f"fdb: reported {len(factors)} factor(s)"
                  + (f" (status={res.get('status')})" if res else ""))
        return res

    def report_ecm(self, factor_fid: str, b1: int, sigma: int | None,
                   b2: int | None = None) -> dict | None:
        """Submit ECM parameters for an already-stored prime factor."""
        params: dict = {"id": factor_fid, "method": "ecm", "param": 3,
                        "b1": str(b1)}
        if sigma is not None:
            params["sigma"] = str(sigma)
        if b2 is not None:
            params["b2"] = str(b2)
        try:
            res = self.rpc("report_ecm", params)
        except FactorDbError as e:
            self._log(f"fdb: report_ecm failed: {e}")
            return None
        ok = bool(isinstance(res, dict) and res.get("ok"))
        self._log(f"fdb: report_ecm {'ok' if ok else 'rejected'}"
                  f" (b1={b1}, sigma={sigma}, b2={b2})")
        return res
