from __future__ import annotations

import random
import itertools
from dataclasses import dataclass, field
from heapq import heappop, heappush
from typing import Dict, List, Tuple, Optional, Union, Set

# ==========================================
# CORE LOGIC & CONSTANTS
# ==========================================

BlockId = int
NodeId = Union[int, str]

EMPTY = "EMPTY"
R1_KFS = "R1_KFS"
R2_REAL = "R2_REAL"
FAKE = "FAKE"

MAX_R1_KFS = 3
MAX_R2_REAL = 4
MAX_FAKE = 1

# Coordinates mapping for the 3x4 Meihua Forest grid
BLOCK_TO_RC: Dict[BlockId, Tuple[int, int]] = {
    1: (0, 0),
    2: (0, 1),
    3: (0, 2),
    4: (1, 0),
    5: (1, 1),
    6: (1, 2),
    7: (2, 0),
    8: (2, 1),
    9: (2, 2),
    10: (3, 0),
    11: (3, 1),
    12: (3, 2),
}

ENTRANCE_BLOCKS: Set[BlockId] = {1, 2, 3}
EXIT_BLOCKS: Set[BlockId] = {10, 11, 12}
VALID_BLOCKS: Set[BlockId] = set(BLOCK_TO_RC.keys())

R1_ALLOWED_BLOCKS: Set[BlockId] = {1, 2, 3, 4, 6, 7, 9, 10, 11, 12}  # Outer ring
R2_ALLOWED_BLOCKS: Set[BlockId] = set(VALID_BLOCKS)
FAKE_ALLOWED_BLOCKS: Set[BlockId] = VALID_BLOCKS - ENTRANCE_BLOCKS

# R1 PERIMETER RING: Strictly defined pathway around the forest
R1_RING = [
    (1, "top"),
    (2, "top"),
    (3, "top"),
    (3, "right"),
    (6, "right"),
    (9, "right"),
    (12, "right"),
    (12, "bottom"),
    (11, "bottom"),
    (10, "bottom"),
    (10, "left"),
    (7, "left"),
    (4, "left"),
    (1, "left"),
]

DEFAULT_HEIGHTS: Dict[BlockId, int] = {
    1: 400,
    2: 200,
    3: 400,
    4: 200,
    5: 400,
    6: 600,
    7: 400,
    8: 600,
    9: 400,
    10: 200,
    11: 400,
    12: 200,
}

OccupancyMap = Dict[BlockId, str]


@dataclass
class BlockInfo:
    rc: Tuple[int, int]
    height_mm: int
    neighbors: List[BlockId] = field(default_factory=list)


@dataclass
class ForestGraph:
    blocks: Dict[BlockId, BlockInfo]

    def neighbors_of(self, block_id: BlockId) -> List[BlockId]:
        return self.blocks[block_id].neighbors

    def height_of(self, block_id: BlockId) -> int:
        return self.blocks[block_id].height_mm


def are_orthogonally_adjacent(a: BlockId, b: BlockId) -> bool:
    ra, ca = BLOCK_TO_RC[a]
    rb, cb = BLOCK_TO_RC[b]
    return abs(ra - rb) + abs(ca - cb) == 1


def is_adjacent_for_picking(a: BlockId, b: BlockId, allow_diagonal: bool) -> bool:
    ra, ca = BLOCK_TO_RC[a]
    rb, cb = BLOCK_TO_RC[b]
    if allow_diagonal:
        return max(abs(ra - rb), abs(ca - cb)) == 1
    return abs(ra - rb) + abs(ca - cb) == 1


def build_forest_graph(block_height_mm: Dict[BlockId, int]) -> ForestGraph:
    blocks: Dict[BlockId, BlockInfo] = {
        b: BlockInfo(rc=rc, height_mm=block_height_mm[b])
        for b, rc in BLOCK_TO_RC.items()
    }
    for a in BLOCK_TO_RC:
        for b in BLOCK_TO_RC:
            if a != b and are_orthogonally_adjacent(a, b):
                blocks[a].neighbors.append(b)
    for b in blocks:
        blocks[b].neighbors.sort()
    return ForestGraph(blocks=blocks)


def edge_cost(a: BlockId, b: BlockId, graph: ForestGraph) -> float:
    return 1.0 + 0.4 * (abs(graph.height_of(a) - graph.height_of(b)) / 200.0)


def path_cost(path: List[BlockId], graph: ForestGraph) -> float:
    if len(path) <= 1:
        return 0.0
    return sum(edge_cost(path[i], path[i + 1], graph) for i in range(len(path) - 1))


def is_physically_clear(block_id: BlockId, graph: ForestGraph) -> bool:
    h = graph.height_of(block_id)
    for nb in graph.neighbors_of(block_id):
        if graph.height_of(nb) - h >= 400:
            return False
    return True


def is_walkable_block(
    block_id: BlockId,
    occupancy: OccupancyMap,
    graph: ForestGraph,
    strict_clearance: bool,
) -> bool:
    if occupancy[block_id] != EMPTY:
        return False
    if strict_clearance and not is_physically_clear(block_id, graph):
        return False
    return True


def legal_pickup_blocks(
    target_block: BlockId,
    occupancy: OccupancyMap,
    graph: ForestGraph,
    allow_diagonal: bool,
    strict_clearance: bool,
) -> List[BlockId]:
    pickup_blocks = []
    for b in VALID_BLOCKS:
        if b != target_block and is_walkable_block(
            b, occupancy, graph, strict_clearance
        ):
            if is_adjacent_for_picking(target_block, b, allow_diagonal):
                pickup_blocks.append(b)
    return pickup_blocks


# --- R2 Routing Logic (Inside Forest) ---
def shortest_legal_path(
    start: NodeId,
    goal: BlockId,
    graph: ForestGraph,
    occupancy: OccupancyMap,
    strict_clearance: bool,
) -> Optional[List[BlockId]]:
    if goal not in graph.blocks or not is_walkable_block(
        goal, occupancy, graph, strict_clearance
    ):
        return None
    dist: Dict[NodeId, float] = {start: 0.0}
    prev: Dict[NodeId, Optional[NodeId]] = {start: None}
    pq: List[Tuple[float, NodeId]] = [(0.0, start)]
    visited: Set[NodeId] = set()

    while pq:
        current_dist, u = heappop(pq)
        if u in visited:
            continue
        visited.add(u)
        if u == goal:
            break

        if u == "ENTRANCE":
            for v in sorted(ENTRANCE_BLOCKS):
                if is_walkable_block(
                    v, occupancy, graph, strict_clearance
                ) and current_dist < dist.get(v, float("inf")):
                    dist[v] = current_dist
                    prev[v] = u
                    heappush(pq, (current_dist, v))
            continue

        for v in graph.neighbors_of(u):  # type: ignore
            if is_walkable_block(v, occupancy, graph, strict_clearance):
                new_dist = current_dist + edge_cost(u, v, graph)  # type: ignore
                if new_dist < dist.get(v, float("inf")):
                    dist[v] = new_dist
                    prev[v] = u
                    heappush(pq, (new_dist, v))

    if goal not in dist:
        return None
    rev_path: List[BlockId] = []
    cur: Optional[NodeId] = goal
    while cur is not None:
        if cur != "ENTRANCE":
            rev_path.append(cur)  # type: ignore
        cur = prev.get(cur)
    rev_path.reverse()
    return rev_path


def best_exit_path(
    start_block: BlockId,
    graph: ForestGraph,
    occupancy: OccupancyMap,
    strict_clearance: bool,
) -> Tuple[Optional[BlockId], Optional[List[BlockId]], float]:
    best_block, best_path, best_cost = None, None, float("inf")
    for exit_block in sorted(EXIT_BLOCKS):
        path = shortest_legal_path(
            start_block, exit_block, graph, occupancy, strict_clearance
        )
        if path is not None and (cost := path_cost(path, graph)) < best_cost:
            best_block, best_path, best_cost = exit_block, path, cost
    return best_block, best_path, best_cost


def make_move_actions_from_path(
    path: List[BlockId], skip_first: bool = False
) -> List[dict]:
    blocks = path[1:] if skip_first else path
    return [{"type": "MOVE", "to": block} for block in blocks]


# --- R1 Perimeter Pathing Logic ---
def get_r1_path(n1: Tuple[int, str], n2: Tuple[int, str]) -> List[Tuple[int, str]]:
    """Calculates the shortest path purely along the defined R1 pathway ring."""
    if n1 == n2:
        return []
    idx1 = R1_RING.index(n1)
    idx2 = R1_RING.index(n2)
    n = len(R1_RING)

    dist_fwd = (idx2 - idx1) % n
    dist_bwd = (idx1 - idx2) % n

    path = []
    if dist_fwd <= dist_bwd:
        curr = idx1
        while curr != idx2:
            curr = (curr + 1) % n
            path.append(R1_RING[curr])
    else:
        curr = idx1
        while curr != idx2:
            curr = (curr - 1) % n
            path.append(R1_RING[curr])
    return path


def build_r1_route_candidates(occupancy: OccupancyMap, n_picks: int) -> List["RoutePlan"]:
    targets = [b for b, v in occupancy.items() if v == R1_KFS]
    if n_picks > len(targets) or n_picks <= 0:
        return []

    candidates = []
    for target_seq in itertools.permutations(targets, n_picks):
        # R1 Spawns at the Top Left Pathway
        current_node = (1, "top")
        cost = 0.0
        acts = []

        for t in target_seq:
            possible_nodes = [n for n in R1_RING if n[0] == t]
            if not possible_nodes:
                break

            best_n, best_path, best_dist = None, [], float("inf")

            for pn in possible_nodes:
                p = get_r1_path(current_node, pn)
                if len(p) < best_dist:
                    best_dist = len(p)
                    best_path = p
                    best_n = pn

            if best_path:
                cost += len(best_path) * 1.0  # 1.0 point per block moved along pathway
                for step in best_path:
                    acts.append({"type": "R1_MOVE", "to": step[0], "side": step[1]})

            cost += 0.75  # Time to pick
            acts.append({"type": "R1_PICK", "target": t, "side": best_n[1]})
            current_node = best_n
        else:
            # R1 Exit Phase - Must reach bottom pathway to exit
            exit_nodes = [(10, "bottom"), (11, "bottom"), (12, "bottom")]
            best_exit, best_ex_path, best_ex_dist = None, [], float("inf")

            for ex in exit_nodes:
                p = get_r1_path(current_node, ex)
                if len(p) < best_ex_dist:
                    best_ex_dist = len(p)
                    best_ex_path = p
                    best_exit = ex

            if best_ex_path:
                cost += len(best_ex_path) * 1.0
                for step in best_ex_path:
                    acts.append({"type": "R1_MOVE", "to": step[0], "side": step[1]})

            acts.append({"type": "R1_EXIT", "via": best_exit[0], "side": best_exit[1]})
            candidates.append(
                RoutePlan(
                    name=f"Perimeter Sequence: {list(target_seq)}",
                    picked_targets=list(target_seq),
                    actions=acts,
                    score=cost,
                    final_block=best_exit[0],
                    exit_block=best_exit[0],
                )
            )

    candidates.sort(key=lambda x: x.score)
    return candidates


@dataclass
class RoutePlan:
    name: str
    picked_targets: List[BlockId]
    actions: List[dict]
    score: float
    final_block: Optional[BlockId]
    exit_block: Optional[BlockId]


def build_n_pick_route_candidates(
    graph: ForestGraph,
    occupancy: OccupancyMap,
    n_picks: int,
    allow_diagonal: bool,
    strict_clearance: bool,
    robot_type: str = "R2",
) -> List[RoutePlan]:
    if robot_type == "R1":
        return build_r1_route_candidates(occupancy, n_picks)

    real_r2s = [b for b, v in occupancy.items() if v == R2_REAL]
    if n_picks > len(real_r2s) or n_picks <= 0:
        return []

    candidates = []
    for target_seq in itertools.permutations(real_r2s, n_picks):
        states = [("ENTRANCE", occupancy, 0.0, [], None)]
        for t in target_seq:
            next_states = []
            for loc, curr_occ, cost, acts, _ in states:
                if loc == "ENTRANCE" and t in ENTRANCE_BLOCKS:
                    new_occ = curr_occ.copy()
                    new_occ[t] = EMPTY
                    next_states.append(
                        (
                            loc,
                            new_occ,
                            cost + 0.75,
                            acts + [{"type": "ENTRANCE_PICK", "target": t}],
                            loc,
                        )
                    )
                for pb in legal_pickup_blocks(
                    t, curr_occ, graph, allow_diagonal, strict_clearance
                ):
                    path = shortest_legal_path(
                        loc, pb, graph, curr_occ, strict_clearance
                    )
                    if path is not None:
                        new_occ = curr_occ.copy()
                        new_occ[t] = EMPTY
                        move_cost = path_cost(path, graph)
                        move_acts = make_move_actions_from_path(
                            path, skip_first=(loc != "ENTRANCE")
                        )
                        next_states.append(
                            (
                                pb,
                                new_occ,
                                cost + move_cost + 0.75,
                                acts
                                + move_acts
                                + [{"type": "PICK_ADJ", "target": t, "from": pb}],
                                pb,
                            )
                        )
            states = next_states
            if not states:
                break

        for loc, curr_occ, cost, acts, _ in states:
            ex_b, ex_path, ex_cost = best_exit_path(
                loc, graph, curr_occ, strict_clearance
            )
            if ex_b is not None and ex_path is not None:
                move_acts = make_move_actions_from_path(ex_path, skip_first=True)
                final_acts = acts + move_acts + [{"type": "EXIT", "via": ex_b}]
                candidates.append(
                    RoutePlan(
                        name=f"Sequence: {list(target_seq)}",
                        picked_targets=list(target_seq),
                        actions=final_acts,
                        score=cost + ex_cost,
                        final_block=loc,
                        exit_block=ex_b,
                    )
                )

    candidates.sort(key=lambda x: x.score)
    return candidates


def generate_random_valid_state() -> Tuple[OccupancyMap, Dict[BlockId, int]]:
    heights = DEFAULT_HEIGHTS.copy()
    occupancy = {b: EMPTY for b in VALID_BLOCKS}
    fake_b = random.choice(list(FAKE_ALLOWED_BLOCKS))
    occupancy[fake_b] = FAKE
    avail_r1 = list(R1_ALLOWED_BLOCKS - {fake_b})
    for b in random.sample(avail_r1, MAX_R1_KFS):
        occupancy[b] = R1_KFS
    avail_r2 = list(
        VALID_BLOCKS - {fake_b} - set([b for b, v in occupancy.items() if v == R1_KFS])
    )
    for b in random.sample(avail_r2, MAX_R2_REAL):
        occupancy[b] = R2_REAL
    return occupancy, heights


def validate_full_setup(
    occupancy: OccupancyMap, heights: Dict[BlockId, int]
) -> Tuple[bool, List[str]]:
    errors: List[str] = []
    r1 = sum(1 for v in occupancy.values() if v == R1_KFS)
    r2 = sum(1 for v in occupancy.values() if v == R2_REAL)
    fk = sum(1 for v in occupancy.values() if v == FAKE)
    if r1 != MAX_R1_KFS:
        errors.append(f"R1 must be {MAX_R1_KFS} (Currently {r1})")
    if r2 != MAX_R2_REAL:
        errors.append(f"R2 must be {MAX_R2_REAL} (Currently {r2})")
    if fk != MAX_FAKE:
        errors.append(f"FAKE must be {MAX_FAKE} (Currently {fk})")
    for b, state in occupancy.items():
        if state == R1_KFS and b not in R1_ALLOWED_BLOCKS:
            errors.append(f"R1 not allowed on block {b}")
        if state == FAKE and b not in FAKE_ALLOWED_BLOCKS:
            errors.append(f"FAKE not allowed on block {b}")
    return len(errors) == 0, errors
