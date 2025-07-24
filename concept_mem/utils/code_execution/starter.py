from dataclasses import dataclass
from typing import List, Tuple, Iterable, Optional
import numpy as np


EIGHT_WAY_DIRECTIONS = [
    (1, 0),
    (-1, 0),
    (0, 1),
    (0, -1),
    (1, 1),
    (1, -1),
    (-1, 1),
    (-1, -1),
]
FOUR_WAY_DIRECTIONS = [(1, 0), (-1, 0), (0, 1), (0, -1)]


@dataclass
class GridObject:
    """
    ONE connected component (4- or 8-way) lifted out of an ARC grid.

    subgrid : np.ndarray  (H_obj × W_obj, dtype uint8)
        The   *tightest bounding-box slice* that contains every pixel of the
        component.  Original colour indices are preserved; any cells inside this
        rectangle that do **not** belong to the component are filled with
        `background_color`.

    position : (int, int)
        (row, col) of the *top-left* corner of `subgrid` inside the parent grid.

    size : int
        Count of foreground pixels in the component (`subgrid != background`).

    colors : List[int]
        **Sorted** list of **all** distinct colour indices present in the
        component (background excluded).  Handles mono- and multi-tone shapes.

    background_color : int
        The value regarded as “empty” inside `subgrid`.  Usually identical to
        the global background, but stored per object so it can be transplanted
        onto grids with different backgrounds later.
    """
    subgrid: np.ndarray
    position: Tuple[int, int]
    size: int
    colors: List[int]
    background_color: int = 0


@dataclass
class GridRegion:
    top_left: tuple[int, int]
    bottom_right: tuple[int, int]

@dataclass
class Pixel:
    """Single cell in a grid."""
    row: int
    col: int
    color: int

@dataclass
class BoundingBox:
    """Axis-aligned rectangle expressed as (min_row, min_col, max_row, max_col)."""
    top: int
    left: int
    bottom: int
    right: int

def extract_components(
    grid: np.ndarray,
    mono_color: bool = True,
    connectivity: int = 8,
    background_color: int = 0,
) -> list[GridObject]:
    """
    Segment a grid into connected components and return them as GridObject-s.

    Parameters
    ----------
    grid : np.ndarray
        H × W array (dtype uint8) with colour indices 0-9.
    mono_color : bool, default True
        • True  – an object contains **one** uniform colour; flood-fill stops
                  when neighbour colour ≠ seed colour.  
        • False – an object may contain many colours; flood-fill joins every
                  non-background pixel that is 4/8-adjacent.
    connectivity : int, {4, 8}
        Pixel-adjacency rule.  Value must be 4 or 8.
    background_color : int, default 0
        Value treated as background.  You may call `detect_background_color`
        beforehand and pass the result here.

    Returns
    -------
    list[GridObject]
        Components in top-to-bottom, left-to-right discovery order.
    """
    if connectivity not in (4, 8):
        raise ValueError("connectivity must be 4 or 8")

    dirs = (
        [(-1, 0), (1, 0), (0, -1), (0, 1)] 
        if connectivity == 4
        else [                              
            (-1, 0), (1, 0), (0, -1), (0, 1),
            (-1, -1), (-1, 1), (1, -1), (1, 1)
        ]
    )

    H, W = grid.shape
    visited = np.zeros((H, W), dtype=bool)
    objects: list[GridObject] = []

    for r in range(H):
        for c in range(W):
            if visited[r, c] or grid[r, c] == background_color:
                continue

            seed_colour = int(grid[r, c])
            stack: list[tuple[int, int]] = [(r, c)]
            visited[r, c] = True
            comp_pix: list[tuple[int, int]] = []

            while stack:
                cr, cc = stack.pop()
                comp_pix.append((cr, cc))
                for dr, dc in dirs:
                    nr, nc = cr + dr, cc + dc
                    if (
                        0 <= nr < H
                        and 0 <= nc < W
                        and not visited[nr, nc]
                        and grid[nr, nc] != background_color
                    ):
                        if mono_color and grid[nr, nc] != seed_colour:
                            continue
                        visited[nr, nc] = True
                        stack.append((nr, nc))
            rows, cols = zip(*comp_pix)
            top, left, bottom, right = min(rows), min(cols), max(rows), max(cols)

            sub = grid[top : bottom + 1, left : right + 1].copy()
            mask_local = np.zeros_like(sub, dtype=bool)
            for pr, pc in comp_pix:
                mask_local[pr - top, pc - left] = True
            sub[~mask_local] = background_color

            if mono_color:
                colours = [seed_colour]
            else:
                colours = sorted(int(col) for col in np.unique(sub[mask_local]))

            objects.append(
                GridObject(
                    subgrid=sub,
                    position=(top, left),
                    size=len(comp_pix),
                    colors=colours,
                    background_color=background_color,
                )
            )

    return objects


def get_component(
    grid: np.ndarray,
    visited: np.ndarray,
    start: tuple[int, int],
    background: int,
    directions: list[tuple[int, int]],
) -> list[tuple[int, int]]:
    """
    General flood-fill that collects all connected pixels (based on given directions)
    which are not equal to the background.
    """
    grid = np.array(grid)
    rows, cols = grid.shape
    stack: list[tuple[int, int]] = [start]
    visited[start] = True
    component: list[tuple[int, int]] = []

    while stack:
        r, c = stack.pop()
        component.append((r, c))
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                if grid[nr, nc] != background and not visited[nr, nc]:
                    visited[nr, nc] = True
                    stack.append((nr, nc))
    return component


#  Color & statistics helpers

def detect_background_color(grid: np.ndarray, ratio: float = 0.33) -> int:
    """
    Heuristically choose the background color.
    If color 0 occupies at least `ratio` of the grid it is returned,
    otherwise the globally most frequent color is assumed.
    """
    if grid.size == 0:
        return 0
    counts = np.bincount(grid.flatten(), minlength=10)
    if counts[0] / grid.size >= ratio:
        return 0
    return int(np.argmax(counts))

def count_colors(grid: np.ndarray) -> np.ndarray:
    """
    Return a length-10 array where index *i* holds the pixel count of color *i*.
    Useful for quick dominance checks or palette extraction.
    """
    return np.bincount(grid.flatten(), minlength=10)


def most_common_non_background(grid: np.ndarray, background: Optional[int] = None) -> int:
    """
    Find the color that appears most frequently excluding the background.
    Background defaults to `detect_background_color(grid)`.
    """
    if background is None:
        background = detect_background_color(grid)
    counts = np.bincount(grid.flatten(), minlength=10)
    counts[background] = -1                     
    return int(np.argmax(counts))


#  Geometry & shape utilities

def bounding_box_of_pixels(coords: Iterable[Tuple[int, int]]) -> BoundingBox:
    """
    Compute the minimal axis-aligned rectangle enclosing a set of (row, col) points.
    """
    coords = list(coords)
    if not coords:
        raise ValueError("Empty coordinate set.")
    rows, cols = zip(*coords)
    return BoundingBox(min(rows), min(cols), max(rows), max(cols))

def crop_to_content(grid: np.ndarray, background: Optional[int] = None) -> np.ndarray:
    """
    Remove all-background rows/columns around the perimeter, returning the tight crop.
    """
    if background is None:
        background = detect_background_color(grid)
    mask = grid != background
    if not mask.any():                           
        return np.array([[background]], dtype=np.uint8)
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    return grid[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]

def reflect_grid(grid: np.ndarray, axis: str = "h") -> np.ndarray:
    """
    Reflect the grid horizontally ('h'), vertically ('v'), or across the main
    diagonals ('d', 'a' for anti-diagonal).  Returns a **new** array.
    """
    if axis == "h":
        return grid[:, ::-1].copy()
    if axis == "v":
        return grid[::-1, :].copy()
    if axis == "d":
        return grid.T.copy()
    if axis == "a":
        return np.flipud(np.fliplr(grid)).T.copy()
    raise ValueError("axis must be one of 'h', 'v', 'd', 'a'")

def rotate_grid(grid: np.ndarray, k: int = 1) -> np.ndarray:
    """
    Rotate the grid 90° *k* times counter-clockwise (k may be negative).
    """
    return np.rot90(grid, k % 4).copy()

def translate_grid(grid: np.ndarray,
                   dr: int,
                   dc: int,
                   fill: Optional[int] = None) -> np.ndarray:
    """
    Shift the whole grid by (dr, dc).  Vacated cells are filled with `fill`
    (defaults to detected background).
    """
    if fill is None:
        fill = detect_background_color(grid)
    out = np.full_like(grid, fill)
    h, w = grid.shape
    rs, cs = np.indices(grid.shape)
    nr, nc = rs + dr, cs + dc
    valid = (0 <= nr) & (nr < h) & (0 <= nc) & (nc < w)
    out[nr[valid], nc[valid]] = grid[rs[valid], cs[valid]]
    return out

# #  Connectivity & component analysis

# _NEIGHBOURS_4 = [(1, 0), (-1, 0), (0, 1), (0, -1)]
# _NEIGHBOURS_8 = _NEIGHBOURS_4 + [(1, 1), (1, -1), (-1, 1), (-1, -1)]

# def connected_components(grid: np.ndarray,
#                          connectivity: int = 8,
#                          background: Optional[int] = None) -> List[np.ndarray]:
#     """
#     Return a list of boolean masks, one per connected component, using
#     4- or 8-connectivity on all non-background pixels.
#     """
#     if connectivity not in (4, 8):
#         raise ValueError("connectivity must be 4 or 8")
#     if background is None:
#         background = detect_background_color(grid)

#     H, W = grid.shape
#     visited = np.zeros((H, W), dtype=bool)
#     neigh = _NEIGHBOURS_4 if connectivity == 4 else _NEIGHBOURS_8
#     comps: List[np.ndarray] = []

#     for r in range(H):
#         for c in range(W):
#             if visited[r, c] or grid[r, c] == background:
#                 continue
#             mask = np.zeros_like(grid, dtype=bool)
#             dq = deque([(r, c)])
#             visited[r, c] = True
#             mask[r, c] = True
#             while dq:
#                 cr, cc = dq.popleft()
#                 for dr, dc in neigh:
#                     nr, nc = cr + dr, cc + dc
#                     if (0 <= nr < H and 0 <= nc < W
#                             and not visited[nr, nc]
#                             and grid[nr, nc] != background):
#                         visited[nr, nc] = True
#                         mask[nr, nc] = True
#                         dq.append((nr, nc))
#             comps.append(mask)
#     return comps


def component_mask_to_pixels(mask: np.ndarray) -> List[Tuple[int, int]]:
    """
    Convert a boolean component mask back to a list of (row, col) coordinates.
    """
    return [tuple(rc) for rc in np.argwhere(mask)]


def split_by_color(mask: np.ndarray,
                   grid: np.ndarray) -> List[np.ndarray]:
    """
    Given a component mask, split it into monochrome sub-masks.
    """
    colours = np.unique(grid[mask])
    return [(mask & (grid == col)) for col in colours]
    
#  Pattern matching & symmetry

def is_symmetric(grid: np.ndarray, axis: str = "h") -> bool:
    """
    Test whether the grid is symmetric about the given axis/diagonal.
    """
    return np.array_equal(grid, reflect_grid(grid, axis))


def match_subshape(small: np.ndarray,
                   big: np.ndarray) -> Optional[Tuple[int, int]]:
    """
    Locate `small` inside `big`.  Returns the (row, col) of the top-left corner
    or None if no exact match exists.
    """
    sh, sw = small.shape
    H, W = big.shape
    for r in range(H - sh + 1):
        for c in range(W - sw + 1):
            if np.array_equal(big[r:r + sh, c:c + sw], small):
                return r, c
    return None

def repeat_pattern(pattern: np.ndarray,
                   out_shape: Tuple[int, int]) -> np.ndarray:
    """
    Tile `pattern` until `out_shape` is reached (cropping any excess).
    """
    ph, pw = pattern.shape
    oh, ow = out_shape
    rep_h = (oh + ph - 1) // ph
    rep_w = (ow + pw - 1) // pw
    tiled = np.tile(pattern, (rep_h, rep_w))
    return tiled[:oh, :ow].copy()
    
#  Painting & overlay helpers

def paint_pixels(grid: np.ndarray,
                 coords: Iterable[Tuple[int, int]],
                 color: int) -> np.ndarray:
    """
    Return a copy of `grid` with the specified coordinates recolored.
    """
    out = grid.copy()
    for r, c in coords:
        if 0 <= r < out.shape[0] and 0 <= c < out.shape[1]:
            out[r, c] = color
    return out

def overlay(grid: np.ndarray,
            patch: np.ndarray,
            top_left: Tuple[int, int],
            transparent: Optional[int] = None) -> np.ndarray:
    """
    Paste `patch` onto `grid` at `top_left`.
    Cells equal to `transparent` in `patch` leave the underlying pixel untouched.
    """
    out = grid.copy()
    tr, tc = top_left
    ph, pw = patch.shape
    for r in range(ph):
        for c in range(pw):
            if 0 <= tr + r < out.shape[0] and 0 <= tc + c < out.shape[1]:
                val = patch[r, c]
                if transparent is None or val != transparent:
                    out[tr + r, tc + c] = val
    return out
    
#  Simple comparison utilities

def grids_equal(a: np.ndarray, b: np.ndarray) -> bool:
    """Exact equality check with identical shape and values."""
    return a.shape == b.shape and np.array_equal(a, b)

def hamming_distance(a: np.ndarray, b: np.ndarray) -> int:
    """
    Count differing cells between two equally-shaped grids
    (helpful for ‘closest match’ heuristics).
    """
    if a.shape != b.shape:
        raise ValueError("grids must share shape for Hamming distance")
    return int((a != b).sum())


from math import floor
from typing import List, Tuple, Optional, Iterable

# GridObject 
# simple geometric queries ---------------------------------

def object_bounding_box(obj: GridObject) -> BoundingBox:
    """Return the object’s bounding box in global coordinates."""
    r0, c0 = obj.position
    h, w = obj.subgrid.shape
    return BoundingBox(r0, c0, r0 + h - 1, c0 + w - 1)


def object_centroid(obj: GridObject) -> Tuple[float, float]:
    """
    Geometric centroid of foreground pixels as (row_f, col_f) in **global** coords.
    """
    rs, cs = np.where(obj.subgrid != obj.background_color)
    if rs.size == 0:  # empty?
        return (obj.position[0] + obj.subgrid.shape[0] / 2.0,
                obj.position[1] + obj.subgrid.shape[1] / 2.0)
    return (obj.position[0] + rs.mean(), obj.position[1] + cs.mean())


def object_mask_global(obj: GridObject,
                       grid_shape: Tuple[int, int]) -> np.ndarray:
    """
    Boolean mask (H×W) marking object pixels at their current location.
    """
    mask = np.zeros(grid_shape, dtype=bool)
    r0, c0 = obj.position
    h, w = obj.subgrid.shape
    mask[r0:r0 + h, c0:c0 + w] = obj.subgrid != obj.background_color
    return mask


def translate_object(obj: GridObject, dr: int, dc: int) -> GridObject:
    """
    Shift by (dr, dc); pixel content unchanged.
    """
    return GridObject(
        subgrid=obj.subgrid.copy(),
        position=(obj.position[0] + dr, obj.position[1] + dc),
        size=obj.size,
        colors=obj.colors.copy(),
        background_color=obj.background_color,
    )


def _np_reflect(arr: np.ndarray, axis: str) -> np.ndarray:
    if axis == "h":
        return arr[:, ::-1]
    if axis == "v":
        return arr[::-1, :]
    if axis == "d":
        return arr.T
    if axis == "a":
        return np.flipud(np.fliplr(arr)).T
    raise ValueError("axis must be one of {'h','v','d','a'}")


def reflect_object(
    obj: GridObject,
    axis: str = "h",
    pivot: Optional[Tuple[int, int]] = None,
) -> GridObject:
    """
    Reflect inside the object’s bounding box (pivot ignored for now unless
    supplied == bbox-centre).  Keeps bbox size, so position unchanged.
    """
    sub_new = _np_reflect(obj.subgrid, axis).copy()
    return GridObject(
        subgrid=sub_new,
        position=obj.position,
        size=obj.size,
        colors=obj.colors.copy(),
        background_color=obj.background_color,
    )


def rotate_object(
    obj: GridObject,
    k: int = 1,
    pivot: Optional[Tuple[int, int]] = None,
) -> GridObject:
    """
    Rotate 90°×k CCW around the local bbox centre.  Position may shift when H≠W.
    """
    k = k % 4
    if k == 0:
        return obj
    sub_new = np.rot90(obj.subgrid, k).copy()
    h_old, w_old = obj.subgrid.shape
    h_new, w_new = sub_new.shape
    r0, c0 = obj.position
    r_cen = r0 + (h_old - 1) / 2.0
    c_cen = c0 + (w_old - 1) / 2.0
    r0_new = int(round(r_cen - (h_new - 1) / 2.0))
    c0_new = int(round(c_cen - (w_new - 1) / 2.0))

    return GridObject(
        subgrid=sub_new,
        position=(r0_new, c0_new),
        size=obj.size,
        colors=obj.colors.copy(),
        background_color=obj.background_color,
    )


def scale_object(
    obj: GridObject,
    factor: int,
) -> GridObject:
    """
    Nearest-neighbour scaling by positive integer factor.
    """
    if factor <= 0:
        raise ValueError("factor must be positive")
    if factor == 1:
        return obj

    sub = obj.subgrid
    sub_scaled = np.kron(sub, np.ones((factor, factor), dtype=sub.dtype))

    return GridObject(
        subgrid=sub_scaled,
        position=obj.position,           
        size=obj.size * factor * factor,
        colors=obj.colors.copy(),
        background_color=obj.background_color,
    )


def _anchor_coord(obj: GridObject, anchor: str) -> Tuple[int, int]:
    """
    Helper to map an anchor keyword to global coordinate.
    """
    bbox = object_bounding_box(obj)
    if anchor == "bbox_topleft":
        return (bbox.top, bbox.left)
    if anchor == "bbox_bottomright":
        return (bbox.bottom, bbox.right)
    if anchor == "bbox_center":
        return ((bbox.top + bbox.bottom) // 2, (bbox.left + bbox.right) // 2)
    if anchor == "centroid":
        cen = object_centroid(obj)
        return (int(round(cen[0])), int(round(cen[1])))
    raise ValueError("unsupported anchor")


def align_object(
    obj: GridObject,
    anchor: str,
    target: Tuple[int, int],
) -> GridObject:
    """
    Translate so that <anchor>(obj) == target (both int coordinates).
    """
    ar, ac = _anchor_coord(obj, anchor)
    dr, dc = target[0] - ar, target[1] - ac
    return translate_object(obj, dr, dc)


def paste_object(
    base_grid: np.ndarray,
    obj: GridObject,
    transparent: Optional[int] = None,
) -> np.ndarray:
    """
    Overlay `obj.subgrid` onto a **copy** of base_grid.
    """
    out = base_grid.copy()
    r0, c0 = obj.position
    h, w = obj.subgrid.shape
    for rr in range(h):
        for cc in range(w):
            val = obj.subgrid[rr, cc]
            if transparent is not None and val == transparent:
                continue
            if 0 <= r0 + rr < out.shape[0] and 0 <= c0 + cc < out.shape[1]:
                out[r0 + rr, c0 + cc] = val
    return out


def compare_objects(
    a: GridObject,
    b: GridObject,
    metric: str = "hamming",
) -> float:
    """
    Distance / similarity between objects.
    """
    metric = metric.lower()
    if metric == "size":
        return abs(a.size - b.size)

    bbox_a = object_bounding_box(a)
    bbox_b = object_bounding_box(b)
    t = min(bbox_a.top, bbox_b.top)
    l = min(bbox_a.left, bbox_b.left)
    btm = max(bbox_a.bottom, bbox_b.bottom)
    rgt = max(bbox_a.right, bbox_b.right)
    H, W = btm - t + 1, rgt - l + 1

    mask_a = object_mask_global(a, (H, W))[bbox_a.top - t : bbox_a.bottom - t + 1,
                                           bbox_a.left - l : bbox_a.right - l + 1]
    mask_b = object_mask_global(b, (H, W))[bbox_b.top - t : bbox_b.bottom - t + 1,
                                           bbox_b.left - l : bbox_b.right - l + 1]

    if metric == "iou":
        inter = np.logical_and(mask_a, mask_b).sum()
        union = np.logical_or(mask_a, mask_b).sum()
        return 1.0 - (inter / union) if union else 0.0  # distance
    # default hamming on masks
    diff = np.logical_xor(mask_a, mask_b).sum()
    return diff


def merge_objects(
    objs: List[GridObject],
) -> GridObject:
    """
    Union of masks; colour indices kept from the *first* non-background pixel
    that occupies each cell (z-order = list order).
    """
    if not objs:
        raise ValueError("merge_objects needs at least one object")

    # Determine union bbox
    tops = [o.position[0] for o in objs]
    lefts = [o.position[1] for o in objs]
    bottoms = [o.position[0] + o.subgrid.shape[0] - 1 for o in objs]
    rights = [o.position[1] + o.subgrid.shape[1] - 1 for o in objs]
    t, l, b, r = min(tops), min(lefts), max(bottoms), max(rights)
    H, W = b - t + 1, r - l + 1
    bg = objs[0].background_color

    sub = np.full((H, W), bg, dtype=np.uint8)
    mask_total = np.zeros_like(sub, dtype=bool)

    for o in objs:
        or0, oc0 = o.position
        for rr in range(o.subgrid.shape[0]):
            for cc in range(o.subgrid.shape[1]):
                val = o.subgrid[rr, cc]
                if val == o.background_color:
                    continue
                gr, gc = or0 + rr - t, oc0 + cc - l
                if not mask_total[gr, gc]:  
                    sub[gr, gc] = val
                    mask_total[gr, gc] = True

    colours = sorted(int(c) for c in np.unique(sub[mask_total]))

    return GridObject(
        subgrid=sub,
        position=(t, l),
        size=int(mask_total.sum()),
        colors=colours,
        background_color=bg,
    )


def split_object_by_color(
    obj: GridObject,
) -> List[GridObject]:
    """
    Partition foreground pixels by colour; ignore connectivity.
    """
    pieces: List[GridObject] = []
    for col in obj.colors:
        mask = (obj.subgrid == col)
        if not mask.any():
            continue
        rs, cs = np.where(mask)
        t, l, b, r = rs.min(), cs.min(), rs.max(), cs.max()
        sub = obj.subgrid[t : b + 1, l : r + 1].copy()
        sub[~mask[t : b + 1, l : r + 1]] = obj.background_color

        pieces.append(
            GridObject(
                subgrid=sub,
                position=(obj.position[0] + t, obj.position[1] + l),
                size=int(mask.sum()),
                colors=[int(col)],
                background_color=obj.background_color,
            )
        )
    return pieces

# --- Utility: Extract function signatures for prompt inclusion ---
import ast
import inspect
from typing import Any

def extract_function_signatures(source: str = None) -> list[dict[str, Any]]:
    """
    Extract all top-level function signatures (name, parameters, docstring) from this file.
    Returns a list of dicts: {name, signature, docstring}
    """
    if source is None:
        import pathlib
        path = pathlib.Path(__file__)
        source = path.read_text()
    tree = ast.parse(source)
    signatures = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            name = node.name
            args = [arg.arg for arg in node.args.args]
            # Build signature string
            sig = f"def {name}({', '.join(args)}):"
            docstring = ast.get_docstring(node)
            signatures.append({
                "name": name,
                "signature": sig,
                "docstring": docstring,
            })
    return signatures