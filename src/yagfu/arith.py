"""Elementary factoring stages: trial division, Pollard rho (Brent),
perfect-power detection, integer roots.

All routines are pure stdlib and work on plain Python ints.
"""

from __future__ import annotations

from math import gcd, isqrt

from .primes import PRIMES_UNDER_256, is_probable_prime

#: YAFU's rho strategy: try the polynomials x^2+3, x^2+2, x^2+1 in this
#: order, and give each one `RHO_MAX_BATCHES` batch-GCD rounds (each batch
#: accumulates 256 differences before one gcd — rhomax=1000 in YAFU counts
#: these batches, not individual squarings).
RHO_POLYNOMIALS = (3, 2, 1)
RHO_MAX_BATCHES = 1000
RHO_BATCH_LEN = 256


def trial_divide(n: int, primes=PRIMES_UNDER_256) -> tuple[list[int], int]:
    """Divide out all `primes`; return (prime factors with multiplicity, cofactor)."""
    factors: list[int] = []
    for p in primes:
        if p * p > n:
            break
        while n % p == 0:
            factors.append(p)
            n //= p
    return factors, n


def iroot(n: int, k: int) -> int:
    """Floor(n ** (1/k)) for n >= 0, k >= 1 (exact integer Newton)."""
    if k == 1 or n < 2:
        return n
    if k == 2:
        return isqrt(n)
    x = 1 << ((n.bit_length() + k - 1) // k + 1)  # guaranteed >= root
    while True:
        y = ((k - 1) * x + n // x ** (k - 1)) // k
        if y >= x:
            break
        x = y
    while x ** k > n:
        x -= 1
    return x


def next_prime(n: int) -> int:
    """Smallest prime > n (upward search; BPSW-classified)."""
    from .primes import is_probable_prime

    if n < 2:
        return 2
    c = n + 1 if n % 2 == 0 else n + 2
    while not is_probable_prime(c):
        c += 2
    return c


def perfect_power(n: int) -> tuple[int, int] | None:
    """If n == b^e with e > 1, return (b, e); else None.

    Probes prime exponents ascending, so e is the smallest prime exponent
    (largest base); the caller recurses on b, which handles nested powers.
    """
    if n < 4:
        return None
    from .primes import _sieve

    for k in _sieve(n.bit_length()):
        b = iroot(n, k)
        if b < 2:
            break
        if b ** k == n:
            return b, k
    return None


def _brent(n: int, c: int, max_batches: int, batch_len: int = RHO_BATCH_LEN) -> int | None:
    """One Brent-rho run on n with f(x) = x^2 + c.

    Returns a nontrivial factor, None if the batch budget ran out, or n if
    the cycle collapsed without isolating a factor (caller should move on).
    """
    f = lambda x: (x * x + c) % n
    g, q = 1, 1
    x = y = 2
    ys = x
    r = 1
    batches = 0
    while g == 1 and batches < max_batches:
        x = y
        for _ in range(r):
            y = f(y)
        k = 0
        while k < r and batches < max_batches and g == 1:
            ys = y
            steps = min(batch_len, r - k)
            for _ in range(steps):
                y = f(y)
                q = q * abs(x - y) % n
            k += steps
            batches += 1
            if q:
                g = gcd(q, n)
        r <<= 1
    if g == 1:
        return None
    if g != n:
        return g
    # q == 0 (mod n): backtrack one step at a time from the last save point
    y = ys
    g = gcd(abs(x - y), n)
    while g == 1:
        y = f(y)
        g = gcd(abs(x - y), n)
    return g  # may still be n: this polynomial failed


def pollard_rho(n: int, max_batches: int = RHO_MAX_BATCHES) -> tuple[list[int], int]:
    """YAFU-style rho sweep over the three polynomials.

    After a factor is found the *same* polynomial continues on the reduced
    cofactor until it fails; only then is the next one tried.  Returns
    (factors found, remaining cofactor).  Factors may themselves be
    composite — the engine recurses.
    """
    factors: list[int] = []
    m = n
    for c in RHO_POLYNOMIALS:
        while m > 1 and not is_probable_prime(m):
            f = _brent(m, c, max_batches)
            if f is None or f == m:
                break
            factors.append(f)
            m //= f
        if m == 1 or is_probable_prime(m):
            break
    return factors, m
