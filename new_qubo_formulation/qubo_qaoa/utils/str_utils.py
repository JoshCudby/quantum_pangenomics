"""Utility functions for generating binary string enumerations.

Primarily used by the parameter-exploration scripts to iterate over all
2^n bitstrings when evaluating exact energy spectra via statevector simulation.
"""
from typing import Generator


def genbin(n, bs='') -> Generator[str, None, None]:
    """Generate all 2^n binary strings of length ``n`` in lexicographic order.

    Uses recursive string concatenation; yields one string at a time without
    materialising the full list in memory.

    Args:
        n: Length of each binary string.
        bs: Partial string built up during recursion.  Callers should leave
            this at its default empty string.

    Yields:
        Binary strings of length ``n``, e.g. for ``n=2``:
        ``"00"``, ``"01"``, ``"10"``, ``"11"``.
    """
    if len(bs) == n:
        yield bs
    else:
        yield from genbin(n, bs + '0')
        yield from genbin(n, bs + '1')