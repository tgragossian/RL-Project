from __future__ import annotations
import heapq
from typing import Dict, Tuple, List

# All edge times below were measured at this movespeed (with boots).
MEASURED_MS = 390.0
RECALL_CHANNEL_TIME = 8.0  # seconds to recall (channel + teleport)

# Raw measurements at MEASURED_MS
RAW_EDGE_TIMES_MEASURED_MS: Dict[Tuple[str, str], float] = {
    ("blue_blue",   "blue_gromp"): 2.00,
    ("blue_gromp",  "top_lane"):   9.49,

    ("blue_blue",   "dragon"):     5.18,  # bot scuttle ≈ dragon pit
    ("dragon",      "mid_lane"):   4.14,

    ("blue_blue",   "blue_wolves"):  6.06,
    ("blue_wolves", "blue_raptors"): 11.03,
    ("blue_raptors","blue_red"):     3.93,
    ("blue_red",    "blue_krugs"):   6.34,
    ("blue_krugs",  "bot_lane"):     10.63,

    ("bot_lane",    "dragon"):       3.00,
    ("blue_raptors","dragon"):       3.00,
    ("blue_raptors","mid_lane"):     5.00,

    ("blue_blue",   "blue_base"):    20.00,
    ("blue_base",   "blue_red"):     19.00,
}

# Make everything undirected: A<->B with same time
EDGE_TIMES_MEASURED_MS: Dict[Tuple[str, str], float] = {}
for (a, b), t in RAW_EDGE_TIMES_MEASURED_MS.items():
    EDGE_TIMES_MEASURED_MS[(a, b)] = t
    EDGE_TIMES_MEASURED_MS[(b, a)] = t


def travel_time_direct(a: str, b: str, ms: float) -> float:
    """
    Travel time *if there is a direct edge* between a and b at movespeed `ms`.
    """
    if a == b:
        return 0.0
    base_time = EDGE_TIMES_MEASURED_MS[(a, b)]  # time at MEASURED_MS
    return base_time * (MEASURED_MS / ms)


def shortest_travel_time(a: str, b: str, ms: float) -> float:
    """
    Shortest-path travel time between a and b over the graph of waypoints.
    Uses Dijkstra on the measured edges. If a == b, returns 0.
    """
    if a == b:
        return 0.0

    # Build adjacency from EDGE_TIMES_MEASURED_MS
    adj: Dict[str, List[Tuple[str, float]]] = {}
    for (u, v), t in EDGE_TIMES_MEASURED_MS.items():
        adj.setdefault(u, []).append((v, t))

    # Dijkstra in terms of measured-time, then scale by ms at the end
    heap: List[Tuple[float, str]] = [(0.0, a)]
    dist: Dict[str, float] = {a: 0.0}

    while heap:
        curr_t, node = heapq.heappop(heap)
        if node == b:
            # curr_t is total time at MEASURED_MS
            return curr_t * (MEASURED_MS / ms)
        if curr_t > dist.get(node, float("inf")):
            continue
        for nxt, edge_t in adj.get(node, []):
            new_t = curr_t + edge_t
            if new_t < dist.get(nxt, float("inf")):
                dist[nxt] = new_t
                heapq.heappush(heap, (new_t, nxt))

    # If unreachable (shouldn't happen if the graph is connected)
    raise ValueError(f"No path from {a} to {b}")


def recall_and_travel_time(
    from_node: str,
    to_node: str,
    ms: float,
    base_node: str = "blue_base",
) -> float:
    """
    Total time for:
      - recalling from `from_node` to fountain (8s), then
      - walking (shortest path) from `base_node` to `to_node`.
    """
    if from_node == base_node:
        # Already at base; just walk
        return shortest_travel_time(base_node, to_node, ms)
    walk_time = shortest_travel_time(base_node, to_node, ms)
    return RECALL_CHANNEL_TIME + walk_time
