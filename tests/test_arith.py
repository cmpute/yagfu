import random

import pytest

from yagfu.arith import _brent, iroot, perfect_power, pollard_rho, trial_divide
from yagfu.primes import PRIMES_UNDER_256, _sieve

gmpy2 = pytest.importorskip("gmpy2")


def test_trial_divide():
    factors, cof = trial_divide(2**5 * 3 * 251**2 * 257)
    assert factors == [2] * 5 + [3] + [251, 251]
    assert cof == 257
    factors, cof = trial_divide(10**30)
    assert factors == [2] * 30 + [5] * 30
    assert cof == 1
    factors, cof = trial_divide(999983 * 999979)  # both primes > 256
    assert factors == []
    assert cof == 999983 * 999979


def test_iroot():
    rng = random.Random(1)
    for _ in range(300):
        k = rng.randint(2, 40)
        n = rng.getrandbits(rng.randint(1, 400))
        r = iroot(n, k)
        assert r**k <= n < (r + 1) ** k
    for k in (2, 3, 5, 7, 11):
        for b in (2, 10, 12345, 10**20 + 1):
            n = b**k
            assert iroot(n, k) == b
            assert iroot(n + 1, k) == b
            assert iroot(n - 1, k) == b - 1


def test_perfect_power():
    # returns the smallest prime exponent; the engine recurses on the base
    def check(n):
        r = perfect_power(n)
        assert r is None or r[0] ** r[1] == n
        return r

    assert check(1024) == (32, 2)      # smallest prime exponent, not (2, 10)
    assert check(3**7) == (3, 7)
    assert check((2**61 - 1) ** 2) == (2**61 - 1, 2)
    assert check((2**61 - 1) ** 6) == ((2**61 - 1) ** 3, 2)
    assert check(6**10) == (6**5, 2)
    assert check(2 * 3 * 5) is None
    assert check(999983 * 999979) is None
    assert check(3) is None
    assert check(4) == (2, 2)


def semiprime(digits_p, digits_q, rng):
    p = gmpy2.next_prime(rng.randrange(10 ** (digits_p - 1), 10**digits_p))
    q = gmpy2.next_prime(rng.randrange(10 ** (digits_q - 1), 10**digits_q))
    return int(p), int(q)


def test_brent_finds_small_factors():
    # 1000 batches * 256 steps ~ 256k squarings: reliably finds factors
    # up to ~11 digits (YAFU's rhomax=1000 has the same reach; bigger
    # factors are ECM's job)
    rng = random.Random(42)
    for dp, dq, batches in ((6, 20, 1000), (8, 24, 1000), (10, 20, 1000),
                            (11, 15, 1000), (12, 18, 5000)):
        p, q = semiprime(dp, dq, rng)
        f = _brent(p * q, 1, batches)
        assert f in (p, q)


def test_rho_pipeline():
    rng = random.Random(7)
    p, q = semiprime(9, 12, rng)
    factors, cof = pollard_rho(p * q)
    assert sorted(factors + ([cof] if cof > 1 else [])) == sorted([p, q])
    # several small factors stacked
    ps = [int(gmpy2.next_prime(rng.randrange(10**7, 10**8))) for _ in range(4)]
    n = 1
    for p in ps:
        n *= p
    factors, cof = pollard_rho(n)
    got = sorted(factors + ([cof] if cof > 1 else []))
    assert got == sorted(ps)


def test_rho_gives_up_on_balanced_semiprime():
    rng = random.Random(3)
    p, q = semiprime(25, 25, rng)
    factors, cof = pollard_rho(p * q, max_batches=20)  # deliberately tiny budget
    assert factors == []
    assert cof == p * q


def test_sieve_consistency():
    assert _sieve(1000)[::-1][:3] == [997, 991, 983]
    assert _sieve(255) == list(PRIMES_UNDER_256)
