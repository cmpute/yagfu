"""yagfu — yet another GPU factorization utility.

Pipeline per composite:  trial division (< 256) -> Pollard rho (x^2+3,
x^2+2, x^2+1, 1000 batch-GCD rounds each) -> GPU ECM (YAFU-style work
ledger, gmp-ecm GPU stage 1 + gpuecm2 GPU stage 2) -> cuda-mpqs (SIQS) or
msieve + cuda-sieve (NFS), chosen by digit count.
"""

__version__ = "0.1.0"
