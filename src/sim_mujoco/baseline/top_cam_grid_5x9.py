from __future__ import annotations

from dataclasses import dataclass

Point3D = tuple[float, float, float]


@dataclass(frozen=True)
class GridRegion:
    name: str
    row: str
    col: int
    top_left: Point3D
    top_right: Point3D
    bottom_left: Point3D
    bottom_right: Point3D

    @property
    def center(self) -> Point3D:
        return _lerp_point(
            _lerp_point(self.top_left, self.top_right, 0.5),
            _lerp_point(self.bottom_left, self.bottom_right, 0.5),
            0.5,
        )

    @property
    def x_min(self) -> float:
        return min(point[0] for point in self.corners)

    @property
    def x_max(self) -> float:
        return max(point[0] for point in self.corners)

    @property
    def y_min(self) -> float:
        return min(point[1] for point in self.corners)

    @property
    def y_max(self) -> float:
        return max(point[1] for point in self.corners)

    @property
    def corners(self) -> tuple[Point3D, Point3D, Point3D, Point3D]:
        return (self.top_left, self.top_right, self.bottom_left, self.bottom_right)

    def point_at(self, row_t: float, col_t: float) -> Point3D:
        top = _lerp_point(self.top_left, self.top_right, col_t)
        bottom = _lerp_point(self.bottom_left, self.bottom_right, col_t)
        return _lerp_point(top, bottom, row_t)


# Floor intersections of the four yellow preview-grid corners for:
#   top_cam, 640x480, fovy=55
#   margins: left=16, top=32, right=16, bottom=0
#
# The camera is perspective-projected and slightly tilted, so the footprint is
# a quadrilateral rather than the old axis-aligned rectangle. Keeping the true
# four corners is essential for outer cells such as a1 and a9: the old y=+/-0.42
# bounds placed their objects outside the rendered top-camera image.
#
# Labels remain:
#   a1 ... a9  (near the robot base / top of the image)
#   ...
#   e1 ... e9  (far from the robot base / bottom of the image)
TOP_LEFT: Point3D = (-0.029519458446004537, -0.3837620659695289, 0.0)
TOP_RIGHT: Point3D = (-0.018286956111947, 0.32962035636142145, 0.0)
BOTTOM_LEFT: Point3D = (0.5079070674376974, -0.38376206596952905, 0.0)
BOTTOM_RIGHT: Point3D = (0.49494648782147743, 0.32962035636142134, 0.0)

ROWS = ("a", "b", "c", "d", "e")
COLS = (1, 2, 3, 4, 5, 6, 7, 8, 9)


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def _lerp_point(a: Point3D, b: Point3D, t: float) -> Point3D:
    return (
        _lerp(a[0], b[0], t),
        _lerp(a[1], b[1], t),
        _lerp(a[2], b[2], t),
    )


def point_at(row_t: float, col_t: float) -> Point3D:
    top = _lerp_point(TOP_LEFT, TOP_RIGHT, col_t)
    bottom = _lerp_point(BOTTOM_LEFT, BOTTOM_RIGHT, col_t)
    return _lerp_point(top, bottom, row_t)


def make_region(row_index: int, col_index: int) -> GridRegion:
    row = ROWS[row_index]
    col = COLS[col_index]
    row_top = row_index / len(ROWS)
    row_bottom = (row_index + 1) / len(ROWS)
    col_left = col_index / len(COLS)
    col_right = (col_index + 1) / len(COLS)
    return GridRegion(
        name=f"{row}{col}",
        row=row,
        col=col,
        top_left=point_at(row_top, col_left),
        top_right=point_at(row_top, col_right),
        bottom_left=point_at(row_bottom, col_left),
        bottom_right=point_at(row_bottom, col_right),
    )


GRID_MATRIX = tuple(
    tuple(make_region(row_index, col_index) for col_index in range(len(COLS)))
    for row_index in range(len(ROWS))
)
GRID_CELLS = {region.name: region for row in GRID_MATRIX for region in row}


def get_region(cell: str) -> GridRegion:
    return GRID_CELLS[cell.strip().lower()]


def get_center(cell: str) -> Point3D:
    return get_region(cell).center
