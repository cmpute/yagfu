import random

import pytest

from yagfu.primes import (
    PRIMES_UNDER_256,
    _is_sprp,
    _strong_lucas_prp,
    is_probable_prime,
    is_proven_prime,
)

gmpy2 = pytest.importorskip("gmpy2")  # reference oracle


def test_small_primes_table():
    assert len(PRIMES_UNDER_256) == 54
    assert PRIMES_UNDER_256[:8] == (2, 3, 5, 7, 11, 13, 17, 19)
    assert PRIMES_UNDER_256[-1] == 251


@pytest.mark.parametrize(
    "n",
    [341, 561, 645, 1105, 1387, 1729, 1905, 2047, 2465, 2701, 2821, 3277,
     4033, 4369, 4371, 4681, 5461, 6601, 7957, 8321, 8481, 8911, 10261,
     13747, 14491, 15709, 18721, 19951, 23377, 31417, 31609, 31621, 49099],
)
def test_classic_pseudoprimes_are_composite(n):
    assert not is_probable_prime(n)


def test_sprp2_pseudoprimes_caught_by_other_bases():
    # 2047 = 23*89 is a strong base-2 pseudoprime; base 7 must catch it.
    assert _is_sprp(2047, 2)
    assert not _is_sprp(2047, 7)


@pytest.mark.parametrize(
    "n",
    [2, 3, 5, 7, 97, 251, 257, 65537, 2147483647, 4294967291],
)
def test_known_primes(n):
    assert is_probable_prime(n)


def test_lucas_primes():
    for n in (1013, 7919, 104729, 15485863, 32452843, 49979687):
        assert _strong_lucas_prp(n)


def test_lucas_composites():
    # odd composites with no factor below 256 (caller's trial division)
    for n in (257 * 263, 1021 * 1031, 15485863 * 32452843, (2**61 - 1) * (2**31 - 1),
              10**36 + 1141 * 1187):
        assert not _strong_lucas_prp(n)


def test_against_gmpy2_random():
    rng = random.Random(20261025)
    # a sweep over many bit sizes, including sizes that use the BPSW path
    for bits in (8, 16, 31, 32, 33, 40, 64, 100, 128, 200, 512, 1024, 2048):
        for _ in range(6):
            n = rng.getrandbits(bits) | 1
            assert is_probable_prime(n) == bool(gmpy2.is_prime(gmpy2.mpz(n), 25)), n


def test_against_gmpy2_structured():
    # numbers near powers, prime powers, and actual primes from gmpy2
    cases = [2**127 - 1, 2**521 - 1, 10**100 + 267, (2**61 - 1) ** 2]
    p = gmpy2.next_prime(10**60)
    cases += [int(p), int(p * gmpy2.next_prime(p)), int(p**2), int(p**3)]
    for n in cases:
        assert is_probable_prime(n) == bool(gmpy2.is_prime(gmpy2.mpz(n), 25)), n


def test_proven_prime_boundary():
    assert is_proven_prime(4759123140) is False
    assert is_proven_prime(4759123141) is False  # 44431 * 107071
    assert is_proven_prime(2**31 - 1) is True    # Mersenne prime, < SPRP bound
    assert is_proven_prime(2**61 - 1) is False   # prime, but only BPSW-known
    assert is_probable_prime(2**61 - 1) is True
