import logging
import math
import os
import shutil
from pathlib import Path

from PIL import Image

import cfg


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def calculate_grid_layout(num_splits):
    """
    Calculate the optimal grid layout for splitting an image into num_splits tiles.
    Returns a list representing the number of tiles in each row, designed to
    center important image elements by putting fewer tiles in outer rows.

    Args:
        num_splits: Total number of subimages desired

    Returns:
        list: Number of tiles per row (e.g., [2, 3] means 2 tiles in row 0, 3 in row 1)
    """
    # Handle small cases explicitly
    if num_splits == 1:
        return [1]
    elif num_splits == 2:
        return [2]
    elif num_splits == 3:
        return [2, 1]
    elif num_splits == 4:
        return [2, 2]
    elif num_splits == 5:
        return [2, 3]
    elif num_splits == 6:
        return [2, 2, 2]
    elif num_splits == 7:
        return [2, 3, 2]
    elif num_splits == 8:
        return [2, 3, 3]
    elif num_splits == 9:
        return [3, 3, 3]
    elif num_splits == 10:
        return [3, 4, 3]
    elif num_splits == 11:
        return [4, 4, 3]
    elif num_splits == 12:
        return [3, 3, 3, 3]
    elif num_splits == 13:
        return [3, 3, 4, 3]
    elif num_splits == 14:
        return [3, 4, 4, 3]
    elif num_splits == 15:
        return [4, 4, 4, 3]
    elif num_splits == 16:
        return [4, 4, 4, 4]
    else:
        # For larger numbers, create a compact, center-focused layout
        # Step 1: Determine the number of rows (aim for square-ish grid)
        rows = round(math.sqrt(num_splits))
        if rows < 3:
            rows = 3

        # Step 2: Calculate average tiles per row
        avg_tiles = num_splits / rows
        base_tiles = int(avg_tiles)
        extra_tiles = num_splits - (base_tiles * rows)

        # Step 3: Distribute tiles across rows
        # Put more tiles in middle rows, fewer on edges
        layout = []
        remaining = num_splits

        for i in range(rows):
            if i == 0 or i == rows - 1:
                # The first and last rows get base or base-1 tiles
                tiles = max(base_tiles - 1, min(base_tiles, remaining))
            else:
                # Middle rows get base or base+1 tiles
                if extra_tiles > 0:
                    tiles = min(base_tiles + 1, remaining)
                    extra_tiles -= 1
                else:
                    tiles = min(base_tiles, remaining)

            if tiles > 0:
                layout.append(tiles)
                remaining -= tiles

        # Adjust if we have remaining tiles (distribute to middle rows)
        while remaining > 0:
            mid_idx = len(layout) // 2
            layout[mid_idx] += 1
            remaining -= 1

        return layout


def split_image_with_overlap(image_path, output_dir, num_splits):
    """
    Split an image into num_splits total subimages with overlap.

    Args:
        image_path: Path to the source image
        output_dir: Directory to save the split images
        num_splits: Total number of subimages (e.g., 3, 4, etc.)
    """
    # Open the image
    img = Image.open(image_path)
    img_width, img_height = img.size

    # Calculate grid layout (returns a list of tiles per row)
    row_layout = calculate_grid_layout(num_splits)
    num_rows = len(row_layout)
    max_cols = max(row_layout)

    # Calculate tile size based on maximum columns
    tile_width = img_width // max_cols
    tile_height = img_height // num_rows

    # Calculate configurable overlap (percentage of tile size)
    overlap_pct = getattr(cfg, 'overlap_percent', 0.25)
    overlap_x = int(tile_width * overlap_pct)
    overlap_y = int(tile_height * overlap_pct)


    # Create an output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Split the image
    tile_index = 0
    for row_idx, tiles_in_row in enumerate(row_layout):
        # Calculate horizontal offset to center tiles in rows with fewer tiles
        offset_cols = (max_cols - tiles_in_row) / 2.0

        for col_idx in range(tiles_in_row):
            # Actual column position (with centering offset)
            actual_col = col_idx + offset_cols

            # Calculate coordinates with overlap
            left = max(0, int(actual_col * tile_width - overlap_x))
            top = max(0, row_idx * tile_height - overlap_y)
            right = min(img_width, int((actual_col + 1) * tile_width + overlap_x))
            bottom = min(img_height, (row_idx + 1) * tile_height + overlap_y)

            # Crop the tile
            tile = img.crop((left, top, right, bottom))

            # Save the tile with naming convention: tile_index.extension
            tile_filename = f"tile_{tile_index}.png"
            tile_path = os.path.join(output_dir, tile_filename)
            tile.save(tile_path)

            tile_index += 1

    logger.info(f"Split {os.path.basename(image_path)} into {num_splits} tiles "
                f"(layout: {row_layout}) in {output_dir}")


def process_all_images(source_image_dir, target_image_dir, num_splits):
    """
    Process all images in source_image_dir and split them into target_image_dir.

    Args:
        source_image_dir: Directory containing source images
        target_image_dir: Directory to save split images (each image gets its own folder)
        num_splits: Total number of subimages (e.g., 3, 4, etc.)
    """
    source_path = Path(source_image_dir)
    target_path = Path(target_image_dir)

    # Remove the target directory if it exists and recreate it
    if os.path.exists(target_path):
        shutil.rmtree(target_path)
        logger.info(f"Removed existing directory: {target_path}")

    # Create a target directory if it doesn't exist
    target_path.mkdir(parents=True, exist_ok=True)

    # Supported image extensions
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}

    # Process each image in the source directory
    for image_file in source_path.iterdir():
        if image_file.is_file() and image_file.suffix.lower() in image_extensions:
            # Create a subdirectory for this image's tiles
            image_name = image_file.stem
            output_dir = target_path / image_name

            try:
                split_image_with_overlap(str(image_file), str(output_dir), num_splits)
            except Exception as e:
                logger.error(f"Error processing {image_file.name}: {e}")


if __name__ == "__main__":
    source_dir = cfg.source_image_dir
    target_dir = cfg.target_image_dir
    number_of_splits = cfg.num_of_splits

    logger.info(f"Starting image splitting...")
    logger.info(f"Source: {source_dir}")
    logger.info(f"Target: {target_dir}")
    logger.info(f"Total subimages per image: {number_of_splits}")

    process_all_images(source_dir, target_dir, number_of_splits)

    logger.info("Image splitting complete!")
