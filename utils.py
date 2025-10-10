"""
Shared utilities for image processing.
"""


def calculate_grid_layout(num_splits):
    """
    Calculate the optimal grid layout for splitting/merging images.
    Returns a list representing the number of tiles in each row, designed to
    center important image elements by putting fewer tiles in outer rows.

    Args:
        num_splits: Total number of subimages (1-10 supported)

    Returns:
        list: Number of tiles per row (e.g., [2, 3] means 2 tiles in row 0, 3 in row 1)

    Raises:
        ValueError: If num_splits is not in the range 1-10
    """
    # Optimized for num_splits 1-10
    layouts = {
        1: [1],
        2: [2],
        3: [1, 2],
        4: [2, 2],
        5: [2, 3],
        6: [2, 2, 2],
        7: [2, 3, 2],
        8: [2, 3, 3],
        9: [3, 3, 3],
        10: [3, 4, 3],
    }
    if num_splits not in layouts:
        raise ValueError(f"num_splits must be between 1 and 10, got {num_splits}")
    return layouts[num_splits]
