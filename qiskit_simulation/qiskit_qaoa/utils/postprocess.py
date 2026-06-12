"""Bitstring filtering using Gosper's hack.

Post-processes QAOA measurement outcomes to enforce a fixed Hamming-weight
constraint (exactly ``T`` ones).  For each sample that has more than ``T``
ones, the full set of bitstrings that can be formed by choosing exactly ``T``
of those one-positions is enumerated using Gosper's hack for iterating over
k-bit combinations.
"""


def postprocess(samples: list[str], T: int) -> list[list[str]]:
    """Filter each sample to all length-T subsets of its set bits.

    For each bitstring in ``samples``, if it has strictly more than ``T`` ones
    (and no more than ``2.5 * T``), produces all bitstrings formed by choosing
    exactly ``T`` of those one-positions and setting the rest to zero.  If the
    sample already has at most ``T`` ones it is returned unchanged.  The
    enumeration uses integer bitmasks and Gosper's hack for efficiency.

    Args:
        samples: A list of binary strings, each of the same length, whose
            entries are ``'0'`` or ``'1'``.
        T: The target Hamming weight.  Samples with exactly ``T`` ones pass
            through unchanged.

    Returns:
        A list (one entry per input sample) of lists of binary strings.
        Each inner list contains the filtered variants of the corresponding
        input sample; if the sample was not filtered it contains the original
        string as a single-element list.
    """
    results = []

    for sample in samples:
        result = []
        n = len(sample)
        # positions of '1' scanning left-to-right: index 0 is leftmost character
        ones_positions = [i for i, ch in enumerate(sample) if ch == '1']
        m = len(ones_positions)

        # only proceed if strictly more than T ones
        if 2.5*T > m > T:
            # Special-case T == 0: produce single all-zero string
            if T == 0:
                result.append('0' * n)
                continue

            # Gosper's hack initialization: iterate all T-bit combinations among m bits
            comb = (1 << T) - 1
            limit = 1 << m

            while comb < limit:
                # Map bits of comb (0..m-1) to actual positions in the full-length sample
                # Build integer mask where bit (n-1-pos) is set (so format(...).zfill(n) yields correct left-to-right order)
                result_mask = 0
                sb = comb
                while sb:
                    lsb = sb & -sb
                    idx = lsb.bit_length() - 1  # index in ones_positions
                    pos = ones_positions[idx]
                    result_mask |= (1 << (n - 1 - pos))
                    sb &= sb - 1

                # Convert to binary string of length n (left-to-right)
                result.append(format(result_mask, 'b').zfill(n))

                # Gosper's next combination
                c = comb & -comb
                r = comb + c
                comb = (((r ^ comb) >> 2) // c) | r
        else:
            result.append(sample)
        results.append(result)
    return results