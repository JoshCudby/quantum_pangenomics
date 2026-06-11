"""Utility for generating binary strings of a fixed width.

Provides a recursive generator that enumerates all ``2^n`` binary strings of
length ``n`` in lexicographic order, used elsewhere in the HUBO pipeline to
iterate over all possible node index encodings.
"""

from typing import Generator


def genbin(n, bs='') -> Generator[str, None, None]:
    """Generate all binary strings of length ``n`` in lexicographic order.

    Recursively appends ``'0'`` and ``'1'`` until the string reaches length ``n``,
    then yields it.  This enumerates all ``2^n`` binary strings without building
    the full list in memory.

    Args:
        n: Desired length of each binary string.
        bs: Prefix accumulated so far during recursion; callers should use the
            default empty string.

    Yields:
        Each binary string of length ``n`` in lexicographic (``'00…0'`` to
        ``'11…1'``) order.
    """
    if len(bs) == n:
        yield bs
    else:
        yield from genbin(n, bs + '0')
        yield from genbin(n, bs + '1')