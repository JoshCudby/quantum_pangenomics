import random
from typing import List, Tuple

def sample_fixed_weight(
    s: str,
    c: int,
    T: int,
    *,
    seed: int | None = None
) -> List[Tuple[str, int]]:
    """
    Given a binary string `s`, a count `c`, and threshold `T`:
      - If s has <= T ones, return the pair (s, c).
      - If s has > T ones, return a list of c pairs [(s_1, 1), ..., (s_c, 1)],
        where each s_i is formed by sampling exactly T positions from the
        1-positions of `s` and setting those positions to '1' (all other
        positions '0'). Each sample is drawn independently (with uniform
        sampling of T positions among the original 1-positions).
    Parameters:
      s: binary string (characters '0' and '1')
      c: integer count (if <= 0 and s has > T ones, returns an empty list)
      T: integer threshold / desired Hamming weight for sampled strings
      seed: optional int seed for reproducible randomness
    Returns:
      - (s, c) if number_of_ones(s) <= T
      - [(s_1, 1), ..., (s_c, 1)] otherwise
    Raises:
      ValueError if s contains characters other than '0' or '1', or if T < 0,
      or if c < 0.
    """
    if any(ch not in '01' for ch in s):
        raise ValueError("s must be a binary string containing only '0' and '1'.")
    if T < 0:
        raise ValueError("T must be non-negative.")
    if c < 0:
        raise ValueError("c must be non-negative.")

    ones = [i for i, ch in enumerate(s) if ch == '1']
    num_ones = len(ones)

    # If s has T or fewer 1s, return (s, c)
    if num_ones <= T:
        return [(s, c)]

    # Otherwise sample c strings each with exactly T ones chosen from 'ones'
    rng = random.Random(seed)
    n = len(s)
    result: List[Tuple[str, int]] = []

    for _ in range(c):
        chosen = rng.sample(ones, T)  # T distinct positions among the original ones
        # Build string with 1s at chosen positions
        chars = ['0'] * n
        for idx in chosen:
            chars[idx] = '1'
        si = ''.join(chars)
        result.append((si, 1))

    return result

def postprocess(samples_dict: dict[str, int], T: int) -> dict[str, int]:
    """
    For each bitstring in `samples`, if it has more than T ones,
    return a list of all bitstrings formed by selecting exactly T of those '1' positions
    (zeros elsewhere). Optimized using integer bitmasks and Gosper's hack.
    """
    results = {}

    for sample, count in samples_dict.items():
        result = sample_fixed_weight(sample, count, T)
            
        for r in result:
            results[r[0]] = results.get(r[0], 0) + r[1]
    return results