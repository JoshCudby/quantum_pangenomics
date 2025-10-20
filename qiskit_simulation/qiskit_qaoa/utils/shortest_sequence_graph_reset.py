from collections import deque
from typing import Dict, List, Set, Tuple, Iterable, Optional

# -----------------------
# GF(2) helpers
# -----------------------
def gf2_rank(rows: List[int], n: int) -> int:
    """Compute rank over GF(2) of matrix with given row bitmasks (rows length n)."""
    A = rows[:]  # copy
    rank = 0
    for col in range(n-1, -1, -1):  # from high bit to low bit
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
# BFS exact solver
# -----------------------
def bfs_shortest_sequence(
    vertices: List,
    edges: Iterable[Tuple],
    labels: Dict,
    max_states: Optional[int] = 5_000_000,
) -> Optional[List[Tuple]]:
    """
    Find a shortest sequence of operations (v, u) meaning L[v] ^= L[u],
    using BFS from the initial label matrix to the identity rows (target).
    Returns list of (v, u) in terms of vertex objects from 'vertices' list, or
    None if impossible / not found (e.g., rank < n or BFS exhausted).
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

    # convert labels to rows (bitmasks)
    rows = labels_to_bitrows(vertices, labels)  # list length n
    # quick rank check
    if gf2_rank(rows, n) < n:
        return None  # impossible

    start = tuple(rows)
    goal = target_identity_rows(n)

    if start == goal:
        return []  # already solved

    q = deque([start])
    parent: Dict[Tuple[int, ...], Tuple[Optional[Tuple[int, ...]], Tuple[int, int]]] = {}
    parent[start] = (None, (-1, -1))  # (prev_state, operation (i,j))

    visited = 1
    while q:
        state = q.popleft()
        cur_rows = list(state)
        for i in range(n):
            for j in adj[i]:
                new_rows = cur_rows[:]  # copy
                new_rows[i] = new_rows[i] ^ new_rows[j]
                new_state = tuple(new_rows)
                if new_state in parent:
                    continue
                parent[new_state] = (state, (i, j))
                if new_state == goal:
                    # reconstruct path
                    ops: List[Tuple[int, int]] = []
                    cur = new_state
                    while parent[cur][0] is not None:
                        prev, op = parent[cur]
                        ops.append(op)
                        cur = prev
                    ops.reverse()
                    # translate to original vertex labels: (i,j) -> (vertices[j], vertices[i])
                    result = [(vertices[j], vertices[i]) for i, j in ops]
                    return result
                q.append(new_state)
                visited += 1
                if max_states is not None and visited > max_states:
                    return None
    return None

# -----------------------
# GF(2) / matrix & elimination helpers
# -----------------------
def labels_to_bitrows(vertices: List, labels: Dict) -> List[int]:
    """Map labels (sets of vertices) to bitmask rows in same vertex index order.
    Convention: bit i corresponds to vertices[i] (LSB=index 0)."""
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

def gaussian_elimination_record_ops(rows: List[int]) -> Tuple[Optional[List[Tuple[str,int,int]]], List[int]]:
    """
    Perform Gaussian elimination to turn rows into identity rows, recording the row operations (over GF(2)).
    This version assigns pivot for column `col` to row `col` (i.e. pivot columns increase 0..n-1),
    ensuring the final matrix is canonical identity with row i == (1 << i).
    Returns (ops, final_rows) where ops is a list of recorded operations of two types:
      - ("xor", i, j) meaning row_i ^= row_j
      - ("swap", i, j) meaning swap row_i and row_j
    If matrix is not full rank, returns (None, rows).
    """
    n = len(rows)
    A = rows[:]  # work copy
    ops: List[Tuple[str,int,int]] = []

    # We'll do elimination so that pivot for column `col` ends up at row `col`.
    # That makes final A equal to identity rows [1<<0, 1<<1, ..., 1<<(n-1)].
    row = 0
    for col in range(0, n):                 # left-to-right: low-bit (col=0) ... high-bit (col=n-1)
        # find a pivot row with bit 'col' set at or below 'row'
        pivot = None
        for r in range(row, n):
            if (A[r] >> col) & 1:
                pivot = r
                break
        if pivot is None:
            continue
        # swap pivot into the target row position if needed
        if pivot != row:
            A[row], A[pivot] = A[pivot], A[row]
            ops.append(("swap", row, pivot))
        # eliminate the bit 'col' from all other rows
        for r in range(n):
            if r != row and ((A[r] >> col) & 1):
                A[r] ^= A[row]
                ops.append(("xor", r, row))
        row += 1
        if row == n:
            break

    if row < n:
        # rank < n, impossible to reach identity
        return None, rows
    # A should now be the identity matrix with row i == (1 << i)
    return ops, A
# -----------------------
# Graph helpers 
# -----------------------
def build_adjacency(vertices: List, edges: Iterable[Tuple]) -> Dict[int, Set[int]]:
    idx = {v:i for i,v in enumerate(vertices)}
    n = len(vertices)
    adj = {i:set() for i in range(n)}
    for a,b in edges:
        if a not in idx or b not in idx:
            raise KeyError("edge references unknown vertex")
        ia, ib = idx[a], idx[b]
        adj[ia].add(ib)
        adj[ib].add(ia)
    return adj

def spanning_tree_bfs(adj: Dict[int, Set[int]], root: int = 0) -> List[Tuple[int,int]]:
    parent = {root: None}
    q = deque([root])
    tree_edges = []
    while q:
        u = q.popleft()
        for v in adj[u]:
            if v not in parent:
                parent[v] = u
                tree_edges.append((u,v))
                q.append(v)
    if len(parent) != len(adj):
        raise ValueError(f"graph not connected: {adj}")
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

# -----------------------
# Local synthesis on path
# -----------------------
def synthesize_op_on_path(
    path: List[int], 
    full_rows: List[int],
    goal_delta: Tuple[int,int],
    max_states: int = 50000
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
        u = path[t]
        v = path[t+1]
        actions.append((u,v))
        actions.append((v,u))

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
        for (u,v) in actions:
            iu = index_in_path[u]
            iv = index_in_path[v]
            new_list = list(s)
            new_list[iu] = new_list[iu] ^ new_list[iv]
            new_state = tuple(new_list)
            if new_state in parent:
                continue
            parent[new_state] = (s, (u,v))
            q.append(new_state)
            states_seen += 1
            if states_seen > max_states:
                return None
    return None

# -----------------------
# Heuristic spanning-tree solver
# -----------------------
def heuristic_spanning_tree_solver(vertices: List, edges: Iterable[Tuple], labels: Dict,
                                   max_local_bfs_states: int = 50000,
                                   spanning_tree_edges: Optional[List[Tuple[int,int]]] = None
                                  ) -> Optional[List[Tuple[int,int]]]:
    n = len(vertices)
    adj = build_adjacency(vertices, edges)
    rows = labels_to_bitrows(vertices, labels)
    if gf2_rank(rows, n) < n:
        return None

    ops, _ = gaussian_elimination_record_ops(rows)
    if ops is None:
        return None

    if spanning_tree_edges is None:
        try:
            tree_edges = spanning_tree_bfs(adj, root=0)
        except ValueError:
            print('Could not find spanning tree')
            return None
    else:
        tree_edges = spanning_tree_edges
    tree_adj = {i:set() for i in range(n)}
    for a,b in tree_edges:
        tree_adj[a].add(b)
        tree_adj[b].add(a)

    simulated_ops: List[Tuple[int,int]] = []
    cur_rows = rows[:]  # list of ints, index by integer node id

    def apply_adj_op(u:int, v:int):
        cur_rows[u] ^= cur_rows[v]
        simulated_ops.append((u, v))

    for rec in ops:
        typ, a, b = rec
        if typ == "swap":
            if b in adj[a]:
                apply_adj_op(a,b)
                apply_adj_op(b,a)
                apply_adj_op(a,b)
                continue
            path = path_in_tree(tree_adj, a, b)
            if not path:
                return None
            P = path
            for t in range(len(P)-1):
                u = P[t]; v = P[t+1]
                apply_adj_op(u, v)
                apply_adj_op(v, u)
                apply_adj_op(u, v)
            for t in range(len(P)-2, -1, -1):
                u = P[t]; v = P[t+1]
                apply_adj_op(u, v)
                apply_adj_op(v, u)
                apply_adj_op(u, v)
            continue

        elif typ == "xor":
            if b in adj[a]:
                apply_adj_op(a, b)
                continue
            path = path_in_tree(tree_adj, a, b)
            if not path:
                return None
            acts = synthesize_op_on_path(path, cur_rows, (a,b), max_states=max_local_bfs_states)
            if acts is None:
                return None
            for (u,v) in acts:
                apply_adj_op(u,v)
            continue
        else:
            return None

    # Final target: row i == e_i where bit i is set
    target_rows = [1 << i for i in range(n)]
    if cur_rows == target_rows:
        # convert simulated_ops (indices) to vertex objects and return
        vertex_ops = [(vertices[v], vertices[u]) for (u,v) in simulated_ops]
        # optional small cleanup: remove immediate double-op cancellations (u,v) followed by (u,v)
        i = 0
        while i < len(vertex_ops) - 1:
            if vertex_ops[i] == vertex_ops[i+1]:
                vertex_ops.pop(i)
                vertex_ops.pop(i)
            else:
                i += 1
        return vertex_ops
    else:
        print(cur_rows)
        print(target_rows)
        # failed to reach identity
        return None


