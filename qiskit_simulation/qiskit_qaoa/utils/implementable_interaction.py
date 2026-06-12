#!/usr/bin/env python3
"""XOR/symmetric-difference subset problem for qubit interaction implementability.

Determines which qubit interactions can be implemented without additional SWAP
gates by solving the following problem: given a collection ``S`` of sets
(representing parity information currently stored in each qubit) and a target
set ``T`` (representing the desired interaction), find a subset ``W`` of at
most ``n`` elements of ``S`` whose symmetric difference equals ``T``.

Adaptive strategy selection:

- If ``m = len(S) <= 40``: meet-in-the-middle (fast and exact).
- Else: solve ``A x = t`` over GF(2) via Gaussian elimination; if the
  nullspace is small, brute-force all nullspace combinations.
- Else if ``n <= 6``: depth-first search enumeration up to ``n`` sets.
- Otherwise: report that the problem is hard (NP-hard in general for
  weight-constrained solutions) and return ``None``.
"""

from typing import List, Set, Optional, Tuple, Dict
import itertools

def compress_universe(sets: List[Set[int]], target: Set[int]):
    """Map all elements across ``sets`` and ``target`` to compact integer indices.

    Args:
        sets: A list of sets of integers (the collection ``S``).
        target: The target set ``T``.

    Returns:
        A tuple ``(masks, tmask, mapping, inv_map)`` where:

        - ``masks``: A list of integer bitmasks, one per set in ``sets``.
        - ``tmask``: The bitmask for ``target``.
        - ``mapping``: A dict mapping original elements to bit positions.
        - ``inv_map``: The inverse of ``mapping``.
    """
    # map element -> index
    elems = set()
    for s in sets:
        elems |= s
    elems |= target
    mapping = {e: i for i, e in enumerate(sorted(elems))}
    inv_map = {i: e for e, i in mapping.items()}
    def to_mask(s):
        mask = 0
        for e in s:
            mask |= 1 << mapping[e]
        return mask
    masks = [to_mask(s) for s in sets]
    tmask = to_mask(target)
    return masks, tmask, mapping, inv_map

def popcount(x: int) -> int:
    """Return the number of set bits (1-bits) in the integer ``x``.

    Args:
        x: A non-negative integer.

    Returns:
        The population count (Hamming weight) of ``x``.
    """
    return x.bit_count()

# Strategy 1: meet-in-the-middle for m <= 40
def meet_in_the_middle_find(masks: List[int], target: int, n: int) -> Optional[List[int]]:
    """Find a subset of at most ``n`` bitmasks whose XOR equals ``target``.

    Uses meet-in-the-middle: enumerates all XOR combinations of the left half
    into a hash map, then matches right-half XOR values against the complement
    ``target ^ left_xor``.

    Args:
        masks: List of integer bitmasks (the ``S`` collection after compression).
        target: The target bitmask.
        n: Maximum subset size allowed.

    Returns:
        A list of indices into ``masks`` forming a subset of size ``<= n``
        whose XOR equals ``target``, or ``None`` if no such subset exists
        within size ``n``.
    """
    m = len(masks)
    half = m // 2
    left_idx = list(range(0, half))
    right_idx = list(range(half, m))

    left_map: Dict[int, int] = {}  # diff_mask -> smallest subset bitmask (representing indices as bits)
    # enumerate left subsets
    for r in range(0, len(left_idx) + 1):
        # iterate combinations of indices of size r
        for combo in itertools.combinations(left_idx, r):
            mask = 0
            for i in combo:
                mask ^= masks[i]
            bitset = 0
            for i in combo:
                bitset |= 1 << i
            prev = left_map.get(mask)
            # store the subset bitset with minimal count
            if prev is None or popcount(prev) > popcount(bitset):
                left_map[mask] = bitset

    # enumerate right subsets and try to match
    # But also store right_map to allow combining smallest sizes
    right_map: Dict[int, int] = {}
    for r in range(0, len(right_idx) + 1):
        for combo in itertools.combinations(right_idx, r):
            mask = 0
            for i in combo:
                mask ^= masks[i]
            bitset = 0
            for i in combo:
                bitset |= 1 << i
            prev = right_map.get(mask)
            if prev is None or popcount(prev) > popcount(bitset):
                right_map[mask] = bitset

    # combine
    for lm, lbit in left_map.items():
        needed = target ^ lm
        if needed in right_map:
            rbit = right_map[needed]
            bits = lbit | rbit
            if popcount(bits) <= n:
                # return indices
                return [i for i in range(len(masks)) if (bits >> i) & 1]
    return None

# Strategy 2: Gaussian elimination over GF(2) to get one solution and nullspace basis
def gf2_solve_and_nullspace(masks: List[int], target: int) -> Tuple[bool, List[int], int, List[int]]:
    """
    Solve A x = t over GF(2) where columns of A are masks (size = universe bits),
    variables x correspond to choosing each set. Returns (solvable, particular_solution (as bitvector over vars),
    rank, nullspace_basis (list of bitvectors over vars)).
    All bitvectors over variables are ints with m bits (bit i => x_i = 1).
    """
    # We'll perform elimination on a matrix with rows = universe bits, columns = variables
    # But easier: we work with rows of length m: for each universe bit position r, row vector is which sets include that element.
    # Build row_vectors: list of ints of length m bits
    if not masks:
        return (target == 0, 0, 0, [])

    universe_size = max(mask.bit_length() for mask in masks + [target])
    m = len(masks)
    row_vectors = []
    row_target = []
    for bit in range(universe_size):
        row = 0
        for j, mask in enumerate(masks):
            if (mask >> bit) & 1:
                row |= 1 << j
        row_vectors.append(row)
        row_target.append((target >> bit) & 1)

    # Gaussian elimination on rows; we want to find columns free variables and a particular solution
    rows = row_vectors[:]  # integers length m
    rhs = row_target[:]
    num_rows = len(rows)
    pivot_col_for_row = [-1] * num_rows
    pivot_row_for_col = [-1] * m
    row = 0
    col = 0
    while row < num_rows and col < m:
        # find pivot row with bit col set
        sel = None
        for r in range(row, num_rows):
            if (rows[r] >> col) & 1:
                sel = r
                break
        if sel is None:
            col += 1
            continue
        # swap sel with current row
        if sel != row:
            rows[row], rows[sel] = rows[sel], rows[row]
            rhs[row], rhs[sel] = rhs[sel], rhs[row]
        pivot_col_for_row[row] = col
        pivot_row_for_col[col] = row
        # eliminate other rows
        for r2 in range(num_rows):
            if r2 != row and ((rows[r2] >> col) & 1):
                rows[r2] ^= rows[row]
                rhs[r2] ^= rhs[row]
        row += 1
        col += 1
    rank = row

    # check consistency: any all-zero row with rhs=1 => no solution
    for r in range(rank, num_rows):
        if rows[r] == 0 and rhs[r] == 1:
            return (False, 0, rank, [])

    # Build one particular solution by setting free vars = 0, solving pivots
    particular = 0
    for c in range(m):
        r = pivot_row_for_col[c]
        if r != -1:
            # variable x_c determined by rhs[r] and other (but free variables set to zero)
            # Since after elimination rows[r] has 1 at col c and zeros in other pivot cols; non-pivot cols correspond to free vars (we set to zero)
            val = rhs[r]
            if val:
                particular |= 1 << c
        else:
            # free var, set 0 in particular
            pass

    # Nullspace basis: for each free column f, construct vector with x_f = 1 and pivot variables set accordingly
    nullspace = []
    for f in range(m):
        if pivot_row_for_col[f] == -1:
            vec = 1 << f
            # for each pivot column, determine its value when free f = 1 and all other frees 0
            for c in range(m):
                r = pivot_row_for_col[c]
                if r != -1:
                    # If rows[r] has bit f set, then x_c should toggle when x_f = 1
                    if (rows[r] >> f) & 1:
                        vec |= 1 << c
            nullspace.append(vec)
    return (True, particular, rank, nullspace)

# Strategy 2a: brute-force nullspace combos if nullity small
def search_nullspace_for_weight(particular: int, nullspace: List[int], n: int) -> Optional[List[int]]:
    f = len(nullspace)
    if f == 0:
        if popcount(particular) <= n:
            return [i for i in range(particular.bit_length()) if (particular >> i) & 1]
        return None
    # threshold for brute force
    MAX_F_BRUTE = 24  # 2^24 ~ 16.7M (upper bound; might be heavy)
    if f <= MAX_F_BRUTE:
        # iterate all combinations of nullspace basis
        for mask in range(1 << f):
            vec = particular
            # apply basis vectors where mask bit set
            mm = mask
            idx = 0
            while mm:
                if mm & 1:
                    vec ^= nullspace[idx]
                mm >>= 1
                idx += 1
            if popcount(vec) <= n:
                return [i for i in range(vec.bit_length()) if (vec >> i) & 1]
        return None
    else:
        return None

# Strategy 3: DFS combinations up to size n for small n or when other strategies fail
def dfs_combinations_find(masks: List[int], target: int, n: int, max_explore: int = 10_000_000) -> Optional[List[int]]:
    m = len(masks)
    visited = set()
    best = None
    explored = 0

    def dfs(start, curr_mask, chosen):
        nonlocal explored, best
        if explored > max_explore:
            return
        explored += 1
        if curr_mask == target and len(chosen) <= n:
            best = chosen.copy()
            raise StopIteration  # early exit with success
        if len(chosen) >= n:
            return
        for i in range(start, m):
            new_mask = curr_mask ^ masks[i]
            key = (i, new_mask, len(chosen)+1)
            if key in visited:
                continue
            visited.add(key)
            chosen.append(i)
            dfs(i+1, new_mask, chosen)
            chosen.pop()
            if best is not None:
                return

    try:
        dfs(0, 0, [])
    except StopIteration:
        return best
    return best

# Top-level function that selects strategy
def find_subset_with_symmetric_difference(S: List[Set[int]], T: Set[int], n: int) -> Optional[List[Set[int]]]:
    """Find a subset of ``S`` of size at most ``n`` whose symmetric difference equals ``T``.

    Applies adaptive strategy selection (meet-in-the-middle, GF(2) Gaussian
    elimination + nullspace search, or DFS) based on the size of ``S`` and
    ``n``.

    Args:
        S: A list of sets of integers (the available parity sets).
        T: The target set (the desired interaction).
        n: Maximum number of sets that may be included in the subset.

    Returns:
        A list of sets from ``S`` (not indices) whose symmetric difference
        equals ``T``, or ``None`` if no such subset of size ``<= n`` exists.
    """
    masks, tmask, mapping, inv_map = compress_universe(S, T)
    m = len(masks)
    # Quick checks
    if tmask == 0:
        # empty target: we can return empty set if allowed, else find subset <= n that XORs to empty (pairs etc.)
        if n >= 0:
            return []
    # Strategy selection
    # If small m, use meet-in-the-middle
    if m <= 40:
        res = meet_in_the_middle_find(masks, tmask, n)
        if res is not None:
            return indices_to_sets(res, S)
    # Else try GF(2) elimination to check solvability and maybe find solution with nullspace search
    solvable, particular, rank, nullspace = gf2_solve_and_nullspace(masks, tmask)
    if not solvable:
        return None

    if popcount(particular) <= n:
        return [i for i in range(len(masks)) if (particular >> i) & 1]
    res = search_nullspace_for_weight(particular, nullspace, n)
    if res is not None:
        return indices_to_sets(res, S)
    # If nullspace large but n small, try DFS over original sets up to n
    if n <= 7:
        res = dfs_combinations_find(masks, tmask, n)
        if res is not None:
            return indices_to_sets(res, S)
    return None

# Helper to return the actual sets (not indices)
def indices_to_sets(indices: List[int], S: List[Set[int]]) -> List[Set[int]]:
    return [S[i] for i in indices]


