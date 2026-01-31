from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Dict, Iterable, Iterator, Optional, Tuple

Pos = Tuple[int, int]
Step = Tuple[int, int]

NEIGHBORS_8: Tuple[Step, ...] = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
)


def chebyshev(a: Pos, b: Pos) -> int:
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


def is_adjacent(a: Pos, b: Pos) -> bool:
    return chebyshev(a, b) <= 1


def iter_in_bounds_neighbors8(
    x: int, y: int, *, width: int, height: int
) -> Iterator[Tuple[int, int, int, int]]:
    """
    Yields (nx, ny, dx, dy) for in-bounds 8-neighbors.
    Neighbor order is stable for deterministic movement.
    """
    for dx, dy in NEIGHBORS_8:
        nx, ny = x + dx, y + dy
        if 0 <= nx < width and 0 <= ny < height:
            yield nx, ny, dx, dy


@dataclass(frozen=True)
class MapStatic:
    """
    Static (turn-invariant) map facts.

    This is designed to be built once in BotPlayer.__init__ from the provided
    `map_copy`, avoiding expensive per-turn scans and deep-copies.
    """

    width: int
    height: int
    walkable: Tuple[Tuple[bool, ...], ...]  # [x][y]
    tiles_by_name: Dict[str, Tuple[Pos, ...]]  # e.g. "SHOP" -> ((x,y), ...)

    @classmethod
    def from_map(cls, m) -> "MapStatic":
        width, height = int(m.width), int(m.height)

        walkable_grid = [[False for _ in range(height)] for _ in range(width)]
        by_name: Dict[str, list[Pos]] = {}

        for x in range(width):
            for y in range(height):
                tile = m.tiles[x][y]
                walkable_grid[x][y] = bool(getattr(tile, "is_walkable", False))
                name = getattr(tile, "tile_name", None)
                if not isinstance(name, str) or not name:
                    name = type(tile).__name__.upper()
                by_name.setdefault(name, []).append((x, y))

        walkable = tuple(
            tuple(walkable_grid[x][y] for y in range(height)) for x in range(width)
        )
        tiles_by_name: Dict[str, Tuple[Pos, ...]] = {k: tuple(v) for k, v in by_name.items()}
        # print(tiles_by_name)
        return cls(width=width, height=height, walkable=walkable, tiles_by_name=tiles_by_name)

    def positions(self, tile_name: str) -> Tuple[Pos, ...]:
        return self.tiles_by_name.get(tile_name, ())

    def is_walkable(self, p: Pos) -> bool:
        x, y = p
        return (
            0 <= x < self.width and 0 <= y < self.height and bool(self.walkable[x][y])
        )

    def nearest_tile(self, tile_name: str, from_pos: Pos) -> Optional[Pos]:
        """
        Fast heuristic: nearest by Chebyshev distance (ignores walls).
        For stage-1 scripts this is usually enough and avoids repeated BFS builds.
        """
        best: Optional[Pos] = None
        best_d = 1_000_000
        for p in self.positions(tile_name):
            d = chebyshev(from_pos, p)
            if d < best_d:
                best, best_d = p, d
        return best

    def adjacent_walkable_cells(self, target: Pos) -> frozenset[Pos]:
        """
        Cells where a bot can stand while interacting with `target`.
        Interactions use Chebyshev distance (<= 1), including diagonals.
        """
        tx, ty = target
        goals = set()
        for nx, ny, _, _ in iter_in_bounds_neighbors8(
            tx, ty, width=self.width, height=self.height
        ):
            if self.walkable[nx][ny]:
                goals.add((nx, ny))
        if self.is_walkable(target):
            goals.add(target)
        return frozenset(goals)


@dataclass(frozen=True)
class DistanceField:
    """
    Multi-source BFS distance-to-goal field over walkable tiles.

    dist[x][y] == 0 for goals, increasing by 1 per step; -1 means unreachable.
    """

    width: int
    height: int
    dist: Tuple[Tuple[int, ...], ...]  # [x][y]

    @staticmethod
    def build(static: MapStatic, goals: Iterable[Pos]) -> "DistanceField":
        width, height = static.width, static.height
        dist = [[-1 for _ in range(height)] for _ in range(width)]
        q: deque[Pos] = deque()

        for gx, gy in goals:
            if 0 <= gx < width and 0 <= gy < height and static.walkable[gx][gy]:
                if dist[gx][gy] == -1:
                    dist[gx][gy] = 0
                    q.append((gx, gy))

        while q:
            x, y = q.popleft()
            base = dist[x][y]
            for nx, ny, _, _ in iter_in_bounds_neighbors8(
                x, y, width=width, height=height
            ):
                if not static.walkable[nx][ny]:
                    continue
                if dist[nx][ny] != -1:
                    continue
                dist[nx][ny] = base + 1
                q.append((nx, ny))

        frozen = tuple(tuple(dist[x][y] for y in range(height)) for x in range(width))
        return DistanceField(width=width, height=height, dist=frozen)

    def get(self, p: Pos) -> int:
        x, y = p
        if not (0 <= x < self.width and 0 <= y < self.height):
            return -1
        return int(self.dist[x][y])

    def best_step_from(self, p: Pos) -> Optional[Step]:
        """
        Returns the best (dx, dy) step to decrease distance, or None if stuck/unreachable.
        """
        x, y = p
        here = self.get((x, y))
        if here <= 0:
            return None

        best: Optional[Step] = None
        best_key: Optional[Tuple[int, int, int, int]] = None

        for nx, ny, dx, dy in iter_in_bounds_neighbors8(
            x, y, width=self.width, height=self.height
        ):
            d = self.get((nx, ny))
            if d == -1:
                continue
            # Prefer lower distance; then prefer non-diagonal; then stable dx/dy.
            key = (d, abs(dx) + abs(dy), dx, dy)
            if best_key is None or key < best_key:
                best_key = key
                best = (dx, dy)

        return best


class Navigator:
    """
    Cached navigation utilities for a single static map.

    Stage-1 goal: reduce per-turn work:
    - pre-index stations (MapStatic)
    - cache BFS distance fields for repeated targets (Navigator)
    """

    def __init__(self, static: MapStatic):
        self.static = static
        self._field_cache: Dict[frozenset[Pos], DistanceField] = {}

    def field_to_goals(self, goals: frozenset[Pos]) -> DistanceField:
        field = self._field_cache.get(goals)
        if field is None:
            field = DistanceField.build(self.static, goals)
            self._field_cache[goals] = field
        return field

    def field_to_adjacent_of(self, target: Pos) -> Tuple[frozenset[Pos], DistanceField]:
        goals = self.static.adjacent_walkable_cells(target)
        return goals, self.field_to_goals(goals)

    def step_towards_adjacent(self, from_pos: Pos, target: Pos) -> Optional[Step]:
        if is_adjacent(from_pos, target):
            return None
        goals, field = self.field_to_adjacent_of(target)
        _ = goals  # kept for debugging/extension; field cache key is goals
        return field.best_step_from(from_pos)


def build_game_cmd(
    *,
    red_bot: str,
    blue_bot: str,
    map_path: str,
    turns: Optional[int] = None,
    render: bool = False,
    replay: Optional[str] = None,
) -> str:
    """
    Convenience for consistent local runs (stage-1 "run/test tools").
    Returns a shell command string.
    """
    parts = [
        "python3",
        "src/game.py",
        "--red",
        red_bot,
        "--blue",
        blue_bot,
        "--map",
        map_path,
    ]
    if turns is not None:
        parts += ["--turns", str(int(turns))]
    if render:
        parts += ["--render"]
    if replay is not None:
        parts += ["--replay", replay]
    return " ".join(parts)
