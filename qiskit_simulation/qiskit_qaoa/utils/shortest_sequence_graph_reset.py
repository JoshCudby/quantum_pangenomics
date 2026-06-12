"""CX-sequence solvers for resetting qubit parity to the identity.

After a chain of CX gates accumulates parity information across qubits, the
parity matrix must be reset to the identity (each qubit again stores only its
own information) before the next chain can begin.  This module provides several
algorithms for finding the shortest such reset sequence:

- ``enumerate_removal_pair_sequences``: generates all valid CX sequences that
  reduce a connected qubit graph by one vertex at a time while maintaining
  connectivity, up to ``max_solutions``.
- ``bfs_shortest_sequence``: BFS over the GF(2) row-operation state space to
  find the globally shortest sequence of adjacency CX operations that transform
  the current parity rows into the identity.
- ``heuristic_spanning_tree_solver``: translates standard Gaussian-elimination
  row operations to adjacency-graph CX operations via a spanning tree, providing
  a fast approximate solution when BFS is too expensive.
- ``sort_by_length``: helper to sort interaction tuples by length for chain
  ordering.

Also provides lower-level GF(2) utilities: ``gf2_rank``, ``labels_to_bitrows``,
``bitrows_to_labels``, ``gaussian_elimination_record_ops``, and BFS/tree path
helpers.
"""

from collections import deque, OrderedDict
from typing import Dict, Set, Iterable, List, Generator, Optional, Tuple



def sort_by_length(items, ascending: bool = True):
    """Sort interaction tuples by length, breaking ties lexicographically.

    Args:
        items: An iterable of tuples (or other sized sequences).
        ascending: If ``True`` (default), shorter tuples come first.

    Returns:
        A sorted list of the input items.
    """
    def key_fn(e):
        return (len(e), tuple(sorted(e)))
    return sorted(items, key=key_fn, reverse=not ascending)


# -----------------------
# Connectivity check
# -----------------------
def _bfs_connected(remaining: Set[int], adj: Dict[int, Set[int]]) -> bool:
    """Return True if the induced subgraph on 'remaining' is connected."""
    if not remaining:
        return True
    start = next(iter(remaining))
    q = deque([start])
    seen = {start}
    while q:
        u = q.popleft()
        for w in adj.get(u, ()):
            if w in remaining and w not in seen:
                seen.add(w)
                q.append(w)
    return len(seen) == len(remaining)


# -----------------------
# Enumerate removal pair sequences
# -----------------------
def enumerate_removal_pair_sequences(
    vertices: Iterable[int],
    edges: Iterable[Iterable[int]],
    stop_at: int = 1,
    order: Optional[Iterable[int]] = None,
    neighbor_order: Optional[Dict[int, Iterable[int]]] = None,
    max_solutions: Optional[int] = None,
) -> Generator[List[Tuple[int, int]], None, None]:
    """Enumerate sequences of (vertex, neighbour) CX pairs that reduce a connected graph.

    At each step, a vertex ``v`` is removed from the remaining set and
    "merged" into a neighbour ``u`` (meaning CX(v, u) is applied: row_u ^=
    row_v).  Only removals that leave the remaining graph connected are
    considered.  The generator yields all valid sequences until the graph is
    reduced to ``stop_at`` vertices or ``max_solutions`` sequences have been
    found.

    Args:
        vertices: The initial vertex set.
        edges: Undirected edges as iterables of two vertex labels.
        stop_at: The target number of remaining vertices.  Must satisfy
            ``1 <= stop_at <= len(vertices)``.
        order: Optional preferred ordering of vertices to try first.
        neighbor_order: Optional per-vertex preferred ordering of neighbours.
        max_solutions: Maximum number of sequences to yield before stopping.
            If ``None``, all sequences are yielded.

    Yields:
        Lists of ``(source, target)`` CX gate tuples in application order.

    Raises:
        ValueError: If ``stop_at`` is out of range or the initial graph is
            not connected.
        KeyError: If an edge references a vertex not in ``vertices``.
    """
    """
    Enumerate sequences of (removed_vertex, neighbor) pairs where removing removed_vertex
    and merging into neighbor keeps the rest of the graph connected.

    Convention: a pair (u, v) means 'source u', 'target v' — consistent with row_v ^= row_u.
    """
    V = list(vertices)
    n = len(V)
    if stop_at < 1 or stop_at > n:
        raise ValueError("stop_at must be between 1 and number of vertices")

    adjacency: Dict[int, Set[int]] = {v: set() for v in V}
    for a, b in edges:
        if a not in adjacency or b not in adjacency:
            raise KeyError("edge references unknown vertex")
        adjacency[a].add(b)
        adjacency[b].add(a)

    if not _bfs_connected(set(V), adjacency):
        # initial graph not connected -> no sequences
        return

    # bounded LRU connectivity cache to avoid blow-up
    CONNECTIVITY_CACHE_LIMIT = 20000
    connectivity_cache: "OrderedDict[frozenset, bool]" = OrderedDict()

    def is_connected(remaining: Set[int]) -> bool:
        key = frozenset(remaining)
        v = connectivity_cache.get(key)
        if v is not None:
            return v
        v = _bfs_connected(remaining, adjacency)
        connectivity_cache[key] = v
        if len(connectivity_cache) > CONNECTIVITY_CACHE_LIMIT:
            connectivity_cache.popitem(last=False)
        return v

    candidate_order = list(order) if order is not None else list(V)
    neighbor_order_map: Dict[int, List[int]] = {}
    if neighbor_order is not None:
        for k, seq in neighbor_order.items():
            neighbor_order_map[k] = list(seq)

    remaining = set(V)
    current_pairs: List[Tuple[int, int]] = []
    solutions_found = 0

    def backtrack(remaining: Set[int], current_pairs: List[Tuple[int, int]]):
        nonlocal solutions_found
        if max_solutions is not None and solutions_found >= max_solutions:
            return
        if len(remaining) == stop_at:
            yield list(current_pairs)
            solutions_found += 1
            return

        for v in candidate_order:
            if v not in remaining:
                continue
            remaining2 = remaining - {v}
            if not is_connected(remaining2):
                continue

            neighbours = neighbor_order_map.get(v, None)
            if neighbours is None:
                neighbours_iter = [u for u in adjacency[v] if u in remaining2]
            else:
                neighbours_iter = [u for u in neighbours if u in remaining2]

            # If no neighbours in remaining2 then can't remove v while preserving connectivity
            if not neighbours_iter:
                continue

            for u in neighbours_iter:
                # record operation as (source=u_removed?, target=neighbor?) careful: original semantics
                # We append (v, u) meaning source=v, target=u -> row_u ^= row_v
                current_pairs.append((v, u))
                yield from backtrack(remaining2, current_pairs)
                current_pairs.pop()
                if max_solutions is not None and solutions_found >= max_solutions:
                    return

    yield from backtrack(remaining, current_pairs)


# -----------------------
# Labels <-> bitrows mapping
# -----------------------
def labels_to_bitrows(vertices: List, labels: Dict) -> List[int]:
    """Convert a per-vertex label dict (sets of vertices) to a list of integer bitmasks.

    Args:
        vertices: Ordered list of vertex labels defining the bit positions.
        labels: A dict mapping each vertex to the set of vertices whose parity
            it currently stores (i.e. the ``currently_stored_info`` dict).

    Returns:
        A list of integers (one per vertex) where bit ``j`` is set if vertex
        ``vertices[j]`` is in ``labels[vertices[i]]``.

    Raises:
        KeyError: If a label contains a vertex not in ``vertices``.
    """
    index = {v: i for i, v in enumerate(vertices)}
    rows = []
    for v in vertices:
        mask = 0
        for u in labels[v]:
            if u not in index:
                raise KeyError(f"Label contains unknown vertex {u}")
            mask |= (1 << index[u])
        rows.append(mask)
    return rows


def bitrows_to_labels(vertices: List, rows: List[int]) -> Dict:
    idx_to_v = {i: v for i, v in enumerate(vertices)}
    out = {}
    n = len(vertices)
    for i, mask in enumerate(rows):
        s = set()
        for b in range(n):
            if (mask >> b) & 1:
                s.add(idx_to_v[b])
        out[idx_to_v[i]] = s
    return out


# -----------------------
# GF(2) helpers
# -----------------------
def gf2_rank(rows: List[int], n: int) -> int:
    """Compute rank over GF(2) of matrix with given row bitmasks (rows length n)."""
    A = rows[:]  # copy
    rank = 0
    for col in range(n - 1, -1, -1):
        pivot = None
        for r in range(rank, len(A)):
            if (A[r] >> col) & 1:
                pivot = r
                break
        if pivot is None:
            continue
        A[rank], A[pivot] = A[pivot], A[rank]
        for r in range(len(A)):
            if r != rank and ((A[r] >> col) & 1):
                A[r] ^= A[rank]
        rank += 1
        if rank == n:
            break
    return rank


def target_identity_rows(n: int) -> Tuple[int, ...]:
    """Return tuple of rows representing identity (row i = e_i)."""
    return tuple(1 << i for i in range(n))


# -----------------------
# BFS exact solver (shortest sequence)
# -----------------------
def bfs_shortest_sequence(
    vertices: List,
    edges: Iterable[Tuple],
    labels: Dict,
    max_states: Optional[int] = 5_000_000,
) -> Optional[List[Tuple]]:
    """
    BFS to find shortest sequence of adjacency ops to convert label-rows into identity rows.
    Operation convention: (u, v) means row_v ^= row_u (source u, target v).
    Returns list[(source_label, target_label)] in terms of vertex labels.
    """
    index = {v: i for i, v in enumerate(vertices)}
    n = len(vertices)
    adj = {i: set() for i in range(n)}
    for a, b in edges:
        if a not in index or b not in index:
            raise KeyError("edge references unknown vertex")
        ia, ib = index[a], index[b]
        adj[ia].add(ib)
        adj[ib].add(ia)

    # convert labels to bitrows
    rows = labels_to_bitrows(vertices, labels)
    if gf2_rank(rows, n) < n:
        return None

    start = tuple(rows)
    goal = tuple(1 << i for i in range(n))
    if start == goal:
        return []

    q = deque([start])
    parent: Dict[Tuple[int, ...], Tuple[Optional[Tuple[int, ...]], Tuple[int, int]]] = {}
    parent[start] = (None, (-1, -1))
    visited = 1

    while q:
        state = q.popleft()
        cur_rows = list(state)
        # iterate over directed adjacency (source u -> target v)
        for u in range(n):
            for v in adj[u]:
                new_rows = cur_rows[:]  # copy
                # apply convention: row_v ^= row_u
                new_rows[v] ^= new_rows[u]
                new_state = tuple(new_rows)
                if new_state in parent:
                    continue
                parent[new_state] = (state, (u, v))
                if new_state == goal:
                    # reconstruct path
                    ops: List[Tuple[int, int]] = []
                    cur = new_state
                    while parent[cur][0] is not None:
                        prev, op = parent[cur]
                        ops.append(op)
                        cur = prev
                    ops.reverse()
                    # translate to original vertex labels: (u, v) -> (vertices[u], vertices[v])
                    result = [(vertices[u], vertices[v]) for (u, v) in ops]
                    return result
                q.append(new_state)
                visited += 1
                if max_states is not None and visited > max_states:
                    return None
    return None


# -----------------------
# Gaussian elimination + record ops
# -----------------------
def gaussian_elimination_record_ops(rows: List[int]) -> Tuple[Optional[List[Tuple[str, int, int]]], List[int]]:
    """
    Gaussian elimination over GF(2). Record operations needed to transform rows into identity.
    ops are:
      - ("swap", i, j)
      - ("xor", i, j) meaning row_i ^= row_j
    Note: we record row operations as standard; caller will map these row ops to adjacency ops.
    """
    n = len(rows)
    A = rows[:]  # work copy
    ops: List[Tuple[str, int, int]] = []

    row = 0
    for col in range(0, n):
        pivot = None
        for r in range(row, n):
            if (A[r] >> col) & 1:
                pivot = r
                break
        if pivot is None:
            continue
        if pivot != row:
            A[row], A[pivot] = A[pivot], A[row]
            ops.append(("swap", row, pivot))
        for r in range(n):
            if r != row and ((A[r] >> col) & 1):
                A[r] ^= A[row]
                ops.append(("xor", r, row))
        row += 1
        if row == n:
            break

    if row < n:
        return None, rows
    return ops, A


# -----------------------
# Path utilities & local synthesis
# -----------------------
def build_adjacency(
    vertices: List[int],
    edges: Iterable[Tuple[int,int]],
    *,
    allow_self_loops: bool = False
) -> Dict[int, Set[int]]:
    """
    Build adjacency mapping index->set(index) where indices correspond to positions in `vertices`.
    """
    if len(vertices) != len(set(vertices)):
        raise ValueError("`vertices` contains duplicate labels; labels must be unique")
    idx = {v: i for i, v in enumerate(vertices)}
    n = len(vertices)
    adj: Dict[int, Set[int]] = {i: set() for i in range(n)}
    for a, b in edges:
        if a not in idx or b not in idx:
            raise KeyError(f"edge references unknown vertex: {(a,b)}")
        ia, ib = idx[a], idx[b]
        if ia == ib and not allow_self_loops:
            continue
        adj[ia].add(ib)
        adj[ib].add(ia)
    return adj


def spanning_tree_bfs(adj: Dict[int, Set[int]], root: Optional[int] = None) -> List[Tuple[int,int]]:
    parent: dict[int, Optional[int]] = {}
    if not adj:
        raise ValueError("adj is empty")
    if root is None or root not in adj:
        root = next(iter(adj))
    parent[root] = None
    q = deque([root])
    tree_edges = []
    while q:
        u = q.popleft()
        for v in adj[u]:
            if v not in parent:
                parent[v] = u
                tree_edges.append((u, v))
                q.append(v)
    if len(parent) != len(adj):
        missing = set(adj.keys()) - set(parent.keys())
        raise ValueError(f"graph not connected: missing vertices {missing}")
    return tree_edges


def path_in_tree(tree_adj: Dict[int, Set[int]], a: int, b: int) -> List[int]:
    q = deque([a])
    prev = {a: None}
    while q:
        u = q.popleft()
        if u == b:
            break
        for w in tree_adj[u]:
            if w not in prev:
                prev[w] = u
                q.append(w)
    if b not in prev:
        return []
    path = []
    cur = b
    while cur is not None:
        path.append(cur)
        cur = prev[cur]
    path.reverse()
    return path


def synthesize_op_on_path(
    path: List[int],
    full_rows: List[int],
    goal_delta: Tuple[int,int],
    max_states: int = 50000
) -> Optional[List[Tuple[int,int]]]:
    """
    Synthesize sequence of adjacency ops along `path` to implement row_v ^= row_u
    where goal_delta is (u, v) in terms of node indices in the full graph.
    We work with indices local to the path and return adjacency ops in global indices (u, v).
    Convention: op (a,b) means row_b ^= row_a.
    """
    index_in_path = {node: i for i, node in enumerate(path)}
    orig = tuple(full_rows[node] for node in path)
    target_idx_in_path = index_in_path[goal_delta[0]]
    source_idx_in_path = index_in_path[goal_delta[1]]
    desired = list(orig)
    # desired: the row at target becomes target ^ source
    desired[target_idx_in_path] = orig[target_idx_in_path] ^ orig[source_idx_in_path]
    desired = tuple(desired)

    if orig == desired:
        return []

    # build adjacency actions (directed along path)
    actions = []
    for t in range(len(path)-1):
        u = path[t]; v = path[t+1]
        # (u, v) and (v, u) are valid adjacency ops here
        actions.append((u, v))
        actions.append((v, u))

    start = orig
    q = deque([start])
    parent = {start: (None, None)}
    states_seen = 1
    while q:
        s = q.popleft()
        if s == desired:
            acts = []
            cur = s
            while parent[cur][0] is not None:
                prev, act = parent[cur]
                acts.append(act)
                cur = prev
            acts.reverse()
            return acts
        for (u, v) in actions:
            iu = index_in_path[u]; iv = index_in_path[v]
            new_list = list(s)
            # apply convention: row_v ^= row_u -> target index iv, source iu
            new_list[iv] ^= new_list[iu]
            new_state = tuple(new_list)
            if new_state in parent:
                continue
            parent[new_state] = (s, (u, v))
            q.append(new_state)
            states_seen += 1
            if states_seen > max_states:
                return None
    return None


# -----------------------
# Heuristic spanning-tree solver (uses gaussian ops translated to adjacency ops)
# -----------------------
def heuristic_spanning_tree_solver(
    vertices: List[int],
    edges: Iterable[Tuple[int, int]],
    labels: Dict[int, set[int]],
    max_local_bfs_states: int = 50_000,
    spanning_tree_edges: Optional[List[Tuple[int,int]]] = None
) -> Optional[List[Tuple[int,int]]]:
    """Find a CX sequence that resets qubit parity to the identity using a spanning tree.

    Translates Gaussian-elimination row operations (swaps and XOR) into
    adjacency-graph CX operations by routing along shortest paths in the
    spanning tree.  Faster than full BFS but may produce longer sequences for
    non-tree edges.

    Args:
        vertices: List of integer qubit labels.
        edges: Undirected edges (pairs of vertex labels) defining the
            adjacency graph for CX operations.
        labels: The current ``currently_stored_info`` dict mapping each vertex
            to the set of vertices whose parity it stores.
        max_local_bfs_states: BFS state budget for synthesising individual
            row operations along a path (default 50 000).
        spanning_tree_edges: Optional pre-computed spanning tree edges.
            If ``None``, a BFS spanning tree rooted at vertex 0 is used.

    Returns:
        A list of ``(source, target)`` CX gate tuples (in vertex-label space)
        that transform ``labels`` to the identity, or ``None`` if the solver
        cannot find a valid sequence.
    """
    n = len(vertices)
    adj = build_adjacency(vertices, edges)
    rows = labels_to_bitrows(vertices, labels)
    if gf2_rank(rows, n) < n:
        return None

    ops_recorded, _ = gaussian_elimination_record_ops(rows)
    if ops_recorded is None:
        return None

    if spanning_tree_edges is None:
        try:
            tree_edges = spanning_tree_bfs(adj, root=0)
        except ValueError:
            return None
    else:
        tree_edges = spanning_tree_edges

    # tree adjacency (indices)
    tree_adj = {i:set() for i in range(n)}
    for a,b in tree_edges:
        tree_adj[a].add(b)
        tree_adj[b].add(a)

    simulated_ops: List[Tuple[int,int]] = []
    cur_rows = rows[:]  # list of ints, index by integer node id

    def apply_adj_op(u:int, v:int):
        # apply adjacency operation: row_v ^= row_u
        cur_rows[v] ^= cur_rows[u]
        simulated_ops.append((u, v))

    for rec in ops_recorded:
        typ, a, b = rec
        if typ == "swap":
            # To simulate a row swap using adjacency ops we perform the standard triple-XOR
            # with adjacency ops along shortest path in tree between a and b.
            if b in adj[a]:
                # direct neighbors: three XORs effect a swap when done appropriately
                apply_adj_op(a, b)
                apply_adj_op(b, a)
                apply_adj_op(a, b)
                continue
            path = path_in_tree(tree_adj, a, b)
            if not path:
                return None
            # For swapping rows along path we perform sequence of local swaps using adjacency ops
            # We'll do the standard approach by moving row bits along path and back
            # first move a's contents to b, then restore others
            # forward sweep
            for t in range(len(path)-1):
                u = path[t]; v = path[t+1]
                apply_adj_op(u, v)
                apply_adj_op(v, u)
                apply_adj_op(u, v)
            # reverse sweep to finish
            for t in range(len(path)-2, -1, -1):
                u = path[t]; v = path[t+1]
                apply_adj_op(u, v)
                apply_adj_op(v, u)
                apply_adj_op(u, v)
            continue

        elif typ == "xor":
            # rec is ("xor", i, j) meaning row_i ^= row_j (in row-operation convention)
            # We need to enact this using adjacency ops; row_i ^= row_j corresponds to
            # source = j, target = i in our (u,v) convention. So we need op (j,i)
            # If j is neighbor of i -> direct adjacency operation (j -> i)
            if b in adj[a]:
                apply_adj_op(b, a)  # row_a ^= row_b  -> op (b, a)
                continue
            # otherwise find path in tree and synthesize
            path = path_in_tree(tree_adj, a, b)
            if not path:
                return None
            # synthesize operation along path to implement row_a ^= row_b
            acts = synthesize_op_on_path(path, cur_rows, (a, b), max_states=max_local_bfs_states)
            if acts is None:
                return None
            for (u,v) in acts:
                apply_adj_op(u, v)
            continue
        else:
            return None

    # Final check: cur_rows must equal identity rows
    target_rows = [1 << i for i in range(n)]
    if cur_rows == target_rows:
        # translate simulated_ops (u, v) -> vertex labels (vertices[u], vertices[v])
        vertex_ops = [(vertices[u], vertices[v]) for (u, v) in simulated_ops]
        # remove adjacent cancellations if any (two identical ops in a row cancel)
        i = 0
        while i < len(vertex_ops) - 1:
            if vertex_ops[i] == vertex_ops[i+1]:
                vertex_ops.pop(i)
                vertex_ops.pop(i)
            else:
                i += 1
        return vertex_ops
    else:
        return None


# -----------------------
# IDDFS path and global sequence syntheses
# -----------------------
def iddfs_synthesize_op_on_path(
    path: List[int],
    full_rows: List[int],
    goal_delta: Tuple[int,int],
    max_states: int = 50000,
    max_depth_limit: Optional[int] = None
) -> Optional[List[Tuple[int,int]]]:
    index_in_path = {node:i for i,node in enumerate(path)}
    orig = tuple(full_rows[node] for node in path)
    target_idx_in_path = index_in_path[goal_delta[0]]
    source_idx_in_path = index_in_path[goal_delta[1]]
    desired = list(orig)
    desired[target_idx_in_path] = orig[target_idx_in_path] ^ orig[source_idx_in_path]
    desired = tuple(desired)

    if orig == desired:
        return []

    actions = []
    for t in range(len(path)-1):
        u = path[t]; v = path[t+1]
        actions.append((u,v))
        actions.append((v,u))

    if max_depth_limit is None:
        max_depth_limit = max(6, len(path) * 4)

    states_seen = 0

    def dfs(state: Tuple[int,...], depth: int, limit: int,
            path_actions: List[Tuple[int,int]], seen: Set[Tuple[int,...]]) -> Optional[List[Tuple[int,int]]]:
        nonlocal states_seen
        if states_seen >= max_states:
            return None
        if state == desired:
            return list(path_actions)
        if depth == limit:
            return None

        def score_act(act):
            u,v = act
            iu = index_in_path[u]; iv = index_in_path[v]
            # applying act will set new value of row_iv; check if it matches desired[iv]
            new_row = state[iv] ^ state[iu]
            return 1 if new_row == desired[iv] else 0

        ordered = sorted(actions, key=lambda a: -score_act(a))
        for (u,v) in ordered:
            iu = index_in_path[u]; iv = index_in_path[v]
            new_list = list(state)
            # apply op: row_v ^= row_u
            new_list[iv] ^= new_list[iu]
            new_state = tuple(new_list)
            states_seen += 1
            if new_state in seen:
                continue
            seen.add(new_state)
            path_actions.append((u,v))
            res = dfs(new_state, depth+1, limit, path_actions, seen)
            if res is not None:
                return res
            path_actions.pop()
            seen.remove(new_state)
            if states_seen >= max_states:
                return None
        return None

    for limit in range(1, max_depth_limit + 1):
        res = dfs(orig, 0, limit, [], {orig})
        if res is not None:
            return res
        if states_seen >= max_states:
            return None
    return None


def iddfs_shortest_sequence(
    vertices: List,
    edges: Iterable[Tuple],
    labels: Dict,
    max_states: Optional[int] = 5_000_000,
    max_depth_limit: Optional[int] = None
) -> Optional[List[Tuple]]:
    index = {v: i for i, v in enumerate(vertices)}
    n = len(vertices)
    adj = {i: set() for i in range(n)}
    for a, b in edges:
        if a not in index or b not in index:
            raise KeyError("edge references unknown vertex")
        ia, ib = index[a], index[b]
        adj[ia].add(ib)
        adj[ib].add(ia)

    rows = labels_to_bitrows(vertices, labels)
    if gf2_rank(rows, n) < n:
        return None

    start = tuple(rows)
    goal = tuple(1 << i for i in range(n))
    if start == goal:
        return []

    # actions: all directed adjacency ops (u->v) meaning row_v ^= row_u
    actions = [(i, j) for i in range(n) for j in adj[i]]

    if max_depth_limit is None:
        max_depth_limit = max(8, n * 4)

    states_seen = 0

    def dfs(state: Tuple[int,...], depth: int, limit: int,
            path_actions: List[Tuple[int,int]], seen: Set[Tuple[int,...]]) -> Optional[List[Tuple[int,int]]]:
        nonlocal states_seen
        if states_seen >= (max_states if max_states is not None else 10**12):
            return None
        if state == goal:
            return list(path_actions)
        if depth == limit:
            return None

        def score_op(op):
            i, j = op
            # applying (i,j) yields new row_j; prefer ops that make row_j equal to identity row (1<<j)
            new_row = state[j] ^ state[i]
            return 1 if new_row == (1 << j) else 0

        ordered = sorted(actions, key=lambda op: -score_op(op))
        for (i, j) in ordered:
            new_list = list(state)
            new_list[j] ^= new_list[i]  # row_j ^= row_i
            new_state = tuple(new_list)
            states_seen += 1
            if new_state in seen:
                continue
            seen.add(new_state)
            path_actions.append((i, j))
            res = dfs(new_state, depth+1, limit, path_actions, seen)
            if res is not None:
                return res
            path_actions.pop()
            seen.remove(new_state)
            if states_seen >= (max_states if max_states is not None else 10**12):
                return None
        return None

    for limit in range(1, max_depth_limit + 1):
        res = dfs(start, 0, limit, [], {start})
        if res is not None:
            return [(vertices[u], vertices[v]) for (u, v) in res]
        if states_seen >= (max_states if max_states is not None else 10**12):
            return None

    return None
