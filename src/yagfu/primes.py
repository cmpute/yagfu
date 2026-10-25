"""Primality testing.

* n < 2^32-class range: deterministic Miller-Rabin (SPRP bases 2, 7, 61),
  exact for all n < 4_759_123_141 — this is a *proof*, not a probability.
* larger n: Baillie–PSW (base-2 strong MR + strong Lucas–Selfridge with
  Selfridge's method-A parameters).  No BPSW counterexample is known.

Pure stdlib; Python's native pow() does the modular exponentiation.
"""

from __future__ import annotations

from math import isqrt


def _sieve(limit: int) -> list[int]:
    flags = bytearray([1]) * (limit + 1)
    flags[0] = flags[1] = 0
    for p in range(2, isqrt(limit) + 1):
        if flags[p]:
            flags[p * p :: p] = bytearray(len(flags[p * p :: p]))
    return [i for i, f in enumerate(flags) if f]


#: All 54 primes below 256 — yagfu's trial-division limit.
PRIMES_UNDER_256: tuple[int, ...] = tuple(_sieve(255))

#: Deterministic SPRP(2,7,61) is exact below this bound.
SPRP_2761_LIMIT = 4_759_123_141


def _is_sprp(n: int, a: int) -> bool:
    """Strong probable-prime test of n to base a (n odd > 2, gcd(a,n)=1)."""
    d = n - 1
    s = ((d & -d).bit_length()) - 1
    d >>= s
    x = pow(a, d, n)
    if x == 1 or x == n - 1:
        return True
    for _ in range(s - 1):
        x = x * x % n
        if x == n - 1:
            return True
        if x == 1:  # stuck at 1: will never hit n-1
            return False
    return False


def _jacobi(a: int, n: int) -> int:
    """Jacobi symbol (a/n) for odd n > 0."""
    a %= n
    result = 1
    while a:
        while a % 2 == 0:
            a //= 2
            if n % 8 in (3, 5):
                result = -result
        a, n = n, a
        if a % 4 == 3 and n % 4 == 3:
            result = -result
        a %= n
    return result if n == 1 else 0


def _strong_lucas_prp(n: int) -> bool:
    """Strong Lucas probable-prime test, Selfridge method-A parameters.

    n must be odd, > 3, and coprime to all small primes (the caller has
    already trial-divided).
    """
    # find first D in 5, -7, 9, -11, ... with Jacobi(D/n) = -1
    D = 5
    while True:
        j = _jacobi(D, n)
        if j == -1:
            break
        if j == 0:
            # gcd(|D|, n) > 1: composite unless n is that small prime itself
            return abs(D) == n
        if abs(D) >= 64:
            # dozens of failures essentially only happen for perfect squares
            r = isqrt(n)
            if r * r == n:
                return False
        D = -D - 2 if D > 0 else -D + 2

    P = 1
    Q = (1 - D) // 4  # D == 1 (mod 4), so Q is an integer

    d = n + 1
    s = 0
    while d % 2 == 0:
        d //= 2
        s += 1

    inv2 = (n + 1) // 2
    # (U_m, V_m, Q^m) starting from m = 0
    U, V, Qk = 0, 2, 1
    for bit in bin(d)[2:]:
        # doubling m -> 2m
        U = U * V % n
        V = (V * V - 2 * Qk) % n
        Qk = Qk * Qk % n
        if bit == "1":
            # 2m -> 2m+1
            U, V = (P * U + V) * inv2 % n, (D * U + P * V) * inv2 % n
            Qk = Qk * Q % n
    # U = U_d, V = V_d, Qk = Q^d   (d * 2^s = n + 1)
    if U == 0 or V == 0:
        return True
    for _ in range(s - 1):
        V = (V * V - 2 * Qk) % n
        if V == 0:
            return True
        Qk = Qk * Qk % n
    return False


def is_probable_prime(n: int) -> bool:
    """True if n is (almost certainly) prime."""
    if n < 2:
        return False
    for p in PRIMES_UNDER_256:
        if n % p == 0:
            return n == p
    if n < SPRP_2761_LIMIT:
        return _is_sprp(n, 2) and _is_sprp(n, 7) and _is_sprp(n, 61)
    return _is_sprp(n, 2) and _strong_lucas_prp(n)


def is_proven_prime(n: int) -> bool:
    """True only when primality is *proven*: small enough for deterministic
    SPRP, or discovered by trial division (callers record that separately).

    Everything larger that passes BPSW is only a probable prime (PRP).
    """
    if n < 2:
        return False
    for p in PRIMES_UNDER_256:
        if n % p == 0:
            return n == p
    return n < SPRP_2761_LIMIT and (
        _is_sprp(n, 2) and _is_sprp(n, 7) and _is_sprp(n, 61)
    )
