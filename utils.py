"""
Shared utilities for image processing.
"""
import logging
import os


logger = logging.getLogger(__name__)


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


def discover_categories(data_dir="data"):
    """
    Discover available categories by scanning the data directory.

    Args:
        data_dir: Path to the data directory

    Returns:
        List of category names (subdirectories in data_dir)
    """
    if not os.path.exists(data_dir):
        return []

    categories = [
        d for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d)) and not d.startswith('.')
    ]
    return sorted(categories)


def select_category(data_dir="data"):
    """
    Prompt the user to select a category from available categories in the data directory.

    Args:
        data_dir: Path to the data directory

    Returns:
        Selected category name

    Raises:
        FileNotFoundError: If no categories are found in the data directory
    """
    categories = discover_categories(data_dir)

    if not categories:
        logger.error(f"No categories found in '{data_dir}' directory. Please create category folders first.")
        raise FileNotFoundError(f"No categories available in '{data_dir}' directory")

    print("Available categories:")
    for idx, category in enumerate(categories, start=1):
        print(f"  {idx}. {category}")

    # Get category selection
    while True:
        choice = input(f"\nSelect category (1-{len(categories)}): ").strip()
        try:
            choice_idx = int(choice)
            if 1 <= choice_idx <= len(categories):
                return categories[choice_idx - 1]
            else:
                print(f"Invalid choice. Please enter a number between 1 and {len(categories)}.")
        except ValueError:
            print("Invalid input. Please enter a number.")
