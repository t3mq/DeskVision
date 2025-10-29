from typing import List, Tuple

Point = Tuple[int, int]
Polygon = List[Point]


def point_in_polygon(point: Point, poly: Polygon) -> bool:
    """Ray casting algorithm. Returns True if the point is inside the polygon (or on the edge)."""
    x, y = point
    inside = False
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        intersects = ((y1 > y) != (y2 > y)) and (
            x < (x2 - x1) * (y - y1) / (y2 - y1 + 1e-9) + x1
        )
        if intersects:
            inside = not inside
    return inside


def bbox_center(bbox: Tuple[int, int, int, int]) -> Point:
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) // 2, (y1 + y2) // 2)


def segment_intersection(p1: Point, p2: Point, q1: Point, q2: Point) -> bool:
    """Return True if segments p1-p2 and q1-q2 intersect."""
    def orient(a, b, c):
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

    def on_seg(a, b, c):
        return (
            min(a[0], b[0]) <= c[0] <= max(a[0], b[0])
            and min(a[1], b[1]) <= c[1] <= max(a[1], b[1])
        )

    o1 = orient(p1, p2, q1)
    o2 = orient(p1, p2, q2)
    o3 = orient(q1, q2, p1)
    o4 = orient(q1, q2, p2)

    if o1 == 0 and on_seg(p1, p2, q1):
        return True
    if o2 == 0 and on_seg(p1, p2, q2):
        return True
    if o3 == 0 and on_seg(q1, q2, p1):
        return True
    if o4 == 0 and on_seg(q1, q2, p2):
        return True

    return (o1 > 0) != (o2 > 0) and (o3 > 0) != (o4 > 0)


class Zone:
    def __init__(self, polygon: Polygon, name: str = "zone"):
        self.polygon = polygon
        self.name = name

    def contains(self, point: Point) -> bool:
        """Return True if the point is inside the polygon."""
        return point_in_polygon(point, self.polygon)


class CountLine:
    def __init__(self, p1: Point, p2: Point, name: str = "line", directional: bool = False):
        self.p1 = p1
        self.p2 = p2
        self.name = name
        self.directional = directional

    def crossed(self, path: List[Point]) -> Tuple[bool, int]:
        """Return (has_crossed, dir_sign): dir_sign in {-1, 0, +1}."""
        if len(path) < 2:
            return False, 0
        crossed = False
        sign = 0
        for a, b in zip(path[:-1], path[1:]):
            if segment_intersection(a, b, self.p1, self.p2):
                crossed = True
                v1 = (self.p2[0] - self.p1[0], self.p2[1] - self.p1[1])
                v2 = (b[0] - a[0], b[1] - a[1])
                cross = v1[0] * v2[1] - v1[1] * v2[0]
                sign = 1 if cross > 0 else (-1 if cross < 0 else 0)
        if self.directional:
            return (crossed and sign != 0), sign
        return crossed, sign
