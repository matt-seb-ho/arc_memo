import logging
from typing import Optional

import arckit
import arckit.vis as vis
import drawsvg
import numpy as np
from datasets import Dataset
from IPython.display import display
from PIL import Image

from concept_mem.constants import MAX_LINE_WIDTH
from concept_mem.data.arc_agi import IOPair, Problem
from concept_mem.data.synthetic_barc import (
    get_problem_from_barc,
    load_barc,
)

# set up logging
logger = logging.getLogger(__name__)


# ------------------------------------------------------------------------------
# IMAGE REPRESENTATIONS
# ------------------------------------------------------------------------------


def visualize_grid(arr: np.ndarray):
    return arckit.vis.draw_grid(arr)


def visualize_problem(problem: Problem, **kwargs):
    task = convert_problem_to_task(problem, "custom", dataset="custom")
    return arckit.vis.draw_task(task, include_test="all", **kwargs)


def visualize_barc_problem(idx: int, dataset: Optional[Dataset] = None, **kwargs):
    if dataset is None:
        dataset = load_barc()
    problem = get_problem_from_barc(dataset, idx)
    visualize_problem(problem, **kwargs)


def display_img_file(f):
    img = Image.open(f)
    display(img)
    img.close()


def draw_io_grids(
    task: arckit.data.Task,
    idx: int,
    add_size: bool = False,
    label: bool = False,
    pixel_scale: int = 40,
    out_file: Optional[str] = None,
) -> drawsvg.Drawing:
    # create individual grids
    arr0, arr1 = task.train[idx]
    labels = ("Input", "Output") if label else (None, None)
    grid_i, origin_i, size_i = vis.draw_grid(
        arr0, add_size=add_size, group=True, label=labels[0]
    )
    grid_o, _, size_o = vis.draw_grid(
        arr1, add_size=add_size, group=True, label=labels[1]
    )
    # combine
    width = size_i[0] + size_o[0]
    height = max(size_i[1], size_o[1])
    combined = drawsvg.Drawing(width, height)
    combined.append(drawsvg.Use(grid_i, -origin_i[0], -origin_i[1]))
    combined.append(drawsvg.Use(grid_o, size_i[0] - origin_i[0], -origin_i[1]))
    combined.set_pixel_scale(pixel_scale)
    # save to file
    if out_file:
        vis.output_drawing(combined, out_file)
    return combined


def convert_problem_to_task(
    p: Problem, id_: str, dataset: Optional[str] = None
) -> arckit.data.Task:
    # convert from list[IOPair] to list[dict]
    train_pairs = [_io_pair_to_dict(pair) for pair in p.train_pairs]
    test_pairs = [_io_pair_to_dict(pair) for pair in p.test_pairs]
    task = arckit.data.Task(
        id=id_,
        train=train_pairs,
        test=test_pairs,
        dataset=dataset,
    )
    return task


def _io_pair_to_dict(pair: IOPair) -> dict:
    # convert to list[list[int]]
    return {"input": pair.x.astype(int).tolist(), "output": pair.y.astype(int).tolist()}


# ------------------------------------------------------------------------------
# STRING REPRESENTATIONS
# ------------------------------------------------------------------------------


def grid_repr(grid: np.ndarray, show_size: bool = True) -> str:
    grid_string = np.array2string(
        grid,
        separator=", ",
        max_line_width=MAX_LINE_WIDTH,
    )
    if show_size:
        grid_string += f"\ngrid size = {grid.shape}"
    return grid_string


def example_repr(grid_pair: tuple, show_size: bool = True) -> str:
    if show_size:
        i_shape = grid_pair[0].shape
        o_shape = grid_pair[1].shape
        return (
            f"Input, size = ({i_shape[0]}, {i_shape[1]}):\n"
            f"{grid_repr(grid_pair[0], show_size=False)}\n"
            f"Output, size = ({o_shape[0]}, {o_shape[1]}):\n"
            f"{grid_repr(grid_pair[1], show_size=False)}"
        )
    else:
        return f"Input:\n{grid_repr(grid_pair[0])}\nOutput:\n{grid_repr(grid_pair[1])}"


def task_repr(task: arckit.data.Task, test: bool = True) -> str:
    lines = []
    for i in range(1, len(task.train) + 1):
        lines.append(f"# Example {i}")
        lines.append(example_repr(task.train[i]))
    if test:
        for i in range(len(task.test)):
            lines.append(f"# Test Example Input {i}")
            lines.append(grid_repr(task.test[i][0]))
    return "\n".join(lines)
