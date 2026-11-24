# SPDX-License-Identifier: GPL-2.0-or-later
"""SNFS polynomial forms for a number dividing x^n +/- y^n.

The construction rules are adapted from Phi Polynomial Lab
(GPL-2.0-or-later), https://phi.kyleaskine.com/candidates.mjs and
aurifeuillian.mjs: generic rounding forms, the standard cyclotomic
constructions and the Aurifeuillian L/M forms, every one checked for a
root modulo the target.  This file carries only the enumeration; yagfu
scores and ranks the candidates with score-poly (backends.nfs).

A GPL-2.0-or-later file inside the MIT-licensed package: the Phi rules
it adapts are GPL.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from itertools import product

@dataclass(frozen=True)
class Candidate:
    coefficients: tuple[int, ...]  # c0, ..., cd
    y0: int
    y1: int
    construction: str

    @property
    def degree(self) -> int:
        return len(self.coefficients) - 1

    def reciprocal(self) -> Candidate:
        coefficients = self.coefficients[::-1]
        if coefficients[-1] < 0:
            coefficients = tuple(-c for c in coefficients)
        return Candidate(coefficients, self.y1, self.y0, self.construction)

    def has_root(self, target: int) -> bool:
        if math.gcd(self.y1, target) != 1:
            return False
        root = (-self.y0 * pow(self.y1, -1, target)) % target
        value = 0
        for coefficient in reversed(self.coefficients):
            value = (value * root + coefficient) % target
        return value == 0


def trial_divide(value: int, limit: int, cap: int) -> tuple[list[tuple[int, int]], int]:
    factors = []
    for divisor in (2, *range(3, limit, 2)):
        if value == 1 or len(factors) == cap:
            break
        exponent = 0
        while value % divisor == 0:
            value //= divisor
            exponent += 1
        if exponent:
            factors.append((divisor, exponent))
    return factors, value


def generic_forms(
    x_factors: list[tuple[int, int]], y_factors: list[tuple[int, int]],
    exponent: int, sign: int, degree: int, limit: int,
) -> list[Candidate] | None:
    # Each choice is a multiplier for (leading coefficient, constant, A, B),
    # where A/B is the rational root before canonicalization.
    options = []
    count = 1
    for on_x_side, factors in ((True, x_factors), (False, y_factors)):
        for base, multiplicity in factors:
            quotient, remainder = divmod(exponent * multiplicity, degree)
            if remainder:
                count *= 2
                if count > limit:
                    return None
            lower = base ** quotient
            local_choices = [(base ** remainder, 1, lower, 1)]
            if remainder:
                local_choices.append((1, base ** (degree - remainder), lower * base, 1))
            if not on_x_side:
                local_choices = [
                    (constant, leading, b, a) for leading, constant, a, b in local_choices
                ]
            options.append(local_choices)

    candidates = []
    for choices in product(*options):
        leading, constant, a, b = (math.prod(column) for column in zip(*choices))
        coefficients = (sign * constant,) + (0,) * (degree - 1) + (leading,)
        candidate = Candidate(coefficients, a, -b, "generic rounding")
        candidates.append(candidate.reciprocal() if constant < leading else candidate)
    return candidates


# In each degree Phi tries only the first matching cyclotomic rule.
CYCLOTOMIC_RULES = {
    4: ((15, "real"), (3, "power"), (5, "power")),
    5: ((11, "real"),),
    6: ((21, "real"), (3, "power"), (7, "power"), (13, "real")),
    8: ((15, "direct"), (3, "power"), (5, "power"), (17, "real")),
    9: ((19, "real"),),
}

# Minimal polynomials of z + 1/z for a primitive q-th root of unity z.
REAL_CYCLOTOMICS = {
    11: (1, 3, -3, -4, 1, 1),
    13: (-1, 3, 6, -4, -5, 1, 1),
    15: (1, 4, -4, -1, 1),
    17: (1, -4, -10, 10, 15, -6, -7, 1, 1),
    19: (1, 5, -10, -20, 15, 21, -7, -8, 1, 1),
    21: (1, -8, 8, 6, -6, -1, 1),
}


def cyclotomic_form(x: int, y: int, exponent: int, sign: int, degree: int) -> Candidate | None:
    for q, mode in CYCLOTOMIC_RULES.get(degree, ()):
        if exponent % q:
            continue
        h = exponent // q
        if mode == "real":
            coefficients = tuple(c * (-sign) ** (degree + i)
                                 for i, c in enumerate(REAL_CYCLOTOMICS[q]))
            a, b = x ** h, y ** h
            return Candidate(coefficients, a * a + b * b, -a * b, "cyclotomic")
        if mode == "direct":
            coefficients = tuple(c * (-sign) ** i for i, c in enumerate(
                (1, -1, 0, 1, -1, 1, 0, -1, 1)))
            return Candidate(coefficients, x ** h, -(y ** h), "cyclotomic")

        stride = degree // (q - 1)
        k, remainder = divmod(h, stride)
        if 2 * remainder > stride:
            k += 1
            remainder -= stride
        a, b = x, y
        if remainder < 0:
            a, b = b, a
            remainder = -remainder
        coefficients = [0] * (degree + 1)
        for i in range(q):
            coefficients[i * stride] = (-sign) ** i * a ** (remainder * i) * b ** (remainder * (q - 1 - i))
        return Candidate(tuple(coefficients), x ** k, -(y ** k), "cyclotomic")
    return None


# Keyed by (squarefree product, degree). Rules are tried in order. Each rule
# contains (divisor of h, representation, coefficients for M before scaling).
AURIFEUILLIAN_RULES = {
    (2, 4): ((3, "natural", (1, 2, 2, 2, 1)), (5, "half", (-4, -12, -6, 2, 1)),
             (1, "binomial", ())),
    (2, 6): ((9, "half", (-8, -24, 36, 4, -12, 0, 1)),
             (7, "half", (8, -32, 16, 20, -10, -2, 1)),
             (3, "natural", (1, 0, 0, 4, 0, 0, 1)), (1, "binomial", ())),
    (2, 8): ((15, "half", (16, -96, -96, 88, 64, -24, -14, 2, 1)),
             (3, "quartic", (1,)), (5, "natural", (1, 2, 2, 0, -4, 0, 2, 2, 1)),
             (1, "binomial", ())),
    (3, 4): ((5, "half", (-9, -18, -6, 3, 1)), (1, "binomial", ())),
    (3, 6): ((7, "half", (-27, 54, 18, -36, -12, 3, 1)),
             (3, "natural", (1, 0, 0, 9, 0, 0, 1)), (1, "binomial", ())),
    (3, 8): ((5, "natural", (1, 3, 6, 9, 9, 9, 6, 3, 1)), (1, "binomial", ())),
    (5, 4): ((3, "half", (25, -50, -10, 5, 1)), (1, "natural", (1, 5, 15, 5, 1))),
    (5, 8): ((3, "natural", (1, 5, 10, 25, 75, 25, 10, 5, 1)), (1, "quartic", (3,))),
    (6, 4): ((1, "natural", (1, 6, 18, 6, 1)),),
    (6, 6): ((3, "half", (216, -648, 324, 36, -36, 0, 1)),),
    (6, 8): ((5, "half", (1296, 5184, 0, -1944, 144, 216, -30, -6, 1)),
             (1, "quartic", (3,))),
    (7, 6): ((3, "half", (343, 0, -392, 196, -14, -7, 1)),
             (1, "natural", (1, 7, 21, 49, 21, 7, 1))),
    (10, 4): ((1, "half", (-100, -100, 10, 10, 1)),),
    (10, 8): ((3, "half", (10000, -40000, 32000, -7000, -1200, 600, -30, -10, 1)),
              (1, "natural", (1, 10, 50, 200, 700, 200, 50, 10, 1))),
    (11, 5): ((1, "half", (-1331, -1331, -363, 0, 11, 1)),),
    (13, 6): ((1, "half", (2197, 2197, -676, -338, 13, 13, 1)),),
    (14, 6): ((1, "half", (-2744, -5488, -3136, -588, 14, 14, 1)),),
    (15, 4): ((1, "half", (-225, 0, 60, 15, 1)),),
    (15, 8): ((1, "natural", (1, 15, 120, 675, 2925, 675, 120, 15, 1)),),
    (17, 8): ((1, "half", (83521, 167042, 78608, 0, -6647, -1156, 17, 17, 1)),),
    (21, 6): ((1, "half", (-9261, -18522, -7938, -882, 84, 21, 1)),),
    (30, 8): ((1, "half", (810000, 1620000, 324000, -81000, -28800, -1800, 210, 30, 1)),),
}


@dataclass(frozen=True)
class AurifeuillianContext:
    x: int
    y: int
    h: int
    a: int
    b: int
    xsqr: int
    ysqr: int


def aurifeuillian_context(x: int, y: int, exponent: int, sign: int) -> AurifeuillianContext | None:
    parts = []
    for base in (x, y):
        factors, remainder = trial_divide(base, 31, 6)
        if remainder != 1:
            return None
        parts.append((math.prod(p for p, e in factors if e % 2),
                      math.prod(p ** (e // 2) for p, e in factors)))
    (a, xsqr), (b, ysqr) = parts
    squarefree_product = a * b
    h, remainder = divmod(exponent, squarefree_product)
    required_sign = -1 if squarefree_product % 4 == 1 else 1
    if remainder or h % 2 == 0 or sign != required_sign:
        return None
    return AurifeuillianContext(x, y, h, a, b, xsqr, ysqr)


def aurifeuillian_forms(context: AurifeuillianContext, degree: int) -> tuple[Candidate, ...]:
    x, y, h = context.x, context.y, context.h
    a, b, scale = context.a, context.b, context.xsqr * context.ysqr
    squarefree_product = a * b
    for divisor, mode, template in AURIFEUILLIAN_RULES.get((squarefree_product, degree), ()):
        if h % divisor == 0:
            break
    else:
        return ()

    k = h // divisor
    sign_stride = 1
    if mode == "half":
        coefficients = template
        y0, y1 = x ** k + y ** k, -((x * y) ** ((k - 1) // 2) * scale)
    elif mode == "natural":
        middle = degree // 2
        coefficients = tuple(c * a ** max(0, i - middle) * b ** max(0, middle - i)
                             for i, c in enumerate(template))
        y0 = x ** ((k - 1) // 2) * context.xsqr
        y1 = -(y ** ((k - 1) // 2) * context.ysqr)
    else:
        stride = 4 if mode == "quartic" else degree
        remainder = (k + stride // 2) % stride - stride // 2
        power = (k - remainder) // stride
        lead, constant = (x, y) if remainder > 0 else (y, x)
        r = abs(remainder)
        values = [0] * (degree + 1)
        if mode == "quartic":
            sign_stride = 2
            middle, = template
            values[::2] = (constant ** 2, squarefree_product * scale * constant,
                           middle * x * y, squarefree_product * scale * lead, lead ** 2)
        else:
            sign_stride = degree // 2
            values[0] = constant ** r
            values[degree // 2] = squarefree_product * scale * (x * y) ** ((r - 1) // 2)
            values[degree] = lead ** r
        coefficients = tuple(values)
        y0, y1 = x ** power, -(y ** power)

    # Switch M to L by negating the variable before any power substitution.
    l_coefficients = tuple(c * (-1) ** ((degree + i) // sign_stride)
                           for i, c in enumerate(coefficients))
    return (Candidate(l_coefficients, y0, y1, "Aurifeuillian L"),
            Candidate(coefficients, y0, y1, "Aurifeuillian M"))


def enumerate_candidates(
    x: int, y: int, exponent: int, sign: int, target: int,
    degrees: tuple[int, ...] = (4, 5, 6, 7, 8, 9), limit: int = 512,
) -> tuple[list[Candidate], list[int]]:
    if min(x, y, exponent, limit) < 1 or target <= 1 or sign not in (-1, 1):
        raise ValueError("Expected positive bases, exponent and limit, N > 1, and sign + or -.")
    if x == y or math.gcd(x, y) != 1:
        raise ValueError("The two bases must be different and coprime.")
    x, y = max(x, y), min(x, y)
    if sign == -1 and exponent % 2 == 0:
        exponent, sign = exponent // 2, 1
    if (pow(x, exponent, target) + sign * pow(y, exponent, target)) % target:
        operator = "+" if sign == 1 else "-"
        raise ValueError(f"N does not divide {x}^{exponent}{operator}{y}^{exponent} after Phi normalization.")
    degrees = sorted(set(degrees))
    if not degrees or any(degree not in range(4, 10) for degree in degrees):
        raise ValueError("Choose one or more degrees from 4 through 9.")

    factors = []
    for base in (x, y):
        small, remainder = trial_divide(base, 10000, 15)
        factors.append(small + ([(remainder, 1)] if remainder > 1 else []))
    x_factors, y_factors = factors
    context = aurifeuillian_context(x, y, exponent, sign)
    candidates, skipped = [], []
    for degree in degrees:
        forms = []
        if context is not None:
            aurif = next((c for c in aurifeuillian_forms(context, degree) if c.has_root(target)), None)
            if aurif is not None:
                forms.append(aurif)
        cyclotomic = cyclotomic_form(x, y, exponent, sign, degree)
        if cyclotomic is not None:
            forms.append(cyclotomic)
        generic = generic_forms(x_factors, y_factors, exponent, sign, degree, limit)
        if generic is None:
            skipped.append(degree)
        else:
            forms.extend(generic)
        seen = set()
        for candidate in forms:
            key = (candidate.coefficients, candidate.y0, candidate.y1)
            if key not in seen and candidate.has_root(target):
                seen.add(key)
                candidates.append(candidate)
    return candidates, skipped


def parse_form(expr: str) -> tuple[int, int, int, int]:
    """'x^n-y^n' or 'x^n+y^n' (y^n optional, then 1) -> (x, y, n, sign),
    sign +1 for a sum and -1 for a difference."""
    m = re.fullmatch(r"\s*(\d+)\s*\^\s*(\d+)\s*([-+])\s*(?:(\d+)\s*\^\s*\d+|1)?\s*",
                     expr)
    if not m:
        raise ValueError(f"{expr!r} is not of the form x^n-y^n or x^n+y^n")
    y = int(m.group(4) or 1)
    return int(m.group(1)), y, int(m.group(2)), 1 if m.group(3) == "+" else -1


def form_cyclotomic(b: int, a: int) -> tuple[int, int, int, int]:
    """The smallest x^n +/- y^n that Phi_a(b) divides, as (x, y, n, sign):
    b^a - 1 for odd a, b^(a/2) + 1 for even a."""
    return (b, 1, a, -1) if a % 2 else (b, 1, a // 2, 1)
