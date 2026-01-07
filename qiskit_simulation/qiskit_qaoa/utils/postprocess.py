def postprocess(samples: list[str], T: int) -> list[list[str]]:
    """
    For each bitstring in `samples`, if it has more than T ones,
    return a list of all bitstrings formed by selecting exactly T of those '1' positions
    (zeros elsewhere). Optimized using integer bitmasks and Gosper's hack.
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