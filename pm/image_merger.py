import glob
import logging
import os
from typing import List, Tuple

import cv2 as cv
import numpy as np

import utils


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def list_image_files(folder: str) -> List[str]:
    extensions = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff", "*.webp")
    files: List[str] = []
    for ext in extensions:
        files.extend(glob.glob(os.path.join(folder, ext)))
    files.sort()
    return files


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def stitch_images_grid(images: List[np.ndarray], grid_layout: List[int]) -> np.ndarray:
    """
    Stitch images arranged in a 2D grid layout.

    Args:
        images: List of BGR images
        grid_layout: List indicating number of tiles per row (e.g., [2, 3, 3])

    Returns:
        Stitched BGR image
    """
    if len(images) == 0:
        raise ValueError("No images to stitch.")
    if len(images) == 1:
        return images[0]

    # Verify we have the right number of images
    expected_count = sum(grid_layout)
    if len(images) != expected_count:
        raise ValueError(f"Expected {expected_count} images for layout {grid_layout}, got {len(images)}")

    # Step 1: Stitch each row horizontally
    row_panoramas = []
    tile_idx = 0

    for row_num, tiles_in_row in enumerate(grid_layout):
        row_images = images[tile_idx:tile_idx + tiles_in_row]

        if len(row_images) == 1:
            row_pano = row_images[0]
        else:
            try:
                row_pano = stitch_images_planar(row_images)
            except RuntimeError as e:
                raise RuntimeError(
                    f"Failed to stitch row {row_num} (tiles {tile_idx} to {tile_idx + tiles_in_row - 1}): {e}"
                )

        row_panoramas.append(row_pano)
        tile_idx += tiles_in_row

    # Step 2: Stitch rows vertically
    if len(row_panoramas) == 1:
        return row_panoramas[0]

    # For vertical stitching, we need to stitch rows sequentially
    # Since rows have vertical overlap, we can use planar stitching
    try:
        final_pano = stitch_images_planar(row_panoramas)
    except RuntimeError as e:
        raise RuntimeError(f"Failed to stitch rows vertically: {e}")

    return final_pano


def detect_and_match(
    img1_gray: np.ndarray,
    img2_gray: np.ndarray,
    n_features: int = 5000,
    ratio_test: float = 0.75,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Detect ORB features and compute matches using Lowe's ratio test.
    Returns matched keypoints coordinates (Nx2) for img1 and img2.
    """
    orb = cv.ORB_create(nfeatures=n_features, fastThreshold=5, scaleFactor=1.2, nlevels=8)  # type: ignore[attr-defined]

    keypoints1, desc1 = orb.detectAndCompute(img1_gray, None)
    keypoints2, desc2 = orb.detectAndCompute(img2_gray, None)

    if desc1 is None or desc2 is None or len(keypoints1) < 4 or len(keypoints2) < 4:
        return np.empty((0, 2), dtype=np.float32), np.empty((0, 2), dtype=np.float32)

    bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=False)
    raw_matches = bf.knnMatch(desc1, desc2, k=2)

    good = []
    for m in raw_matches:
        if len(m) == 2:
            m1, m2 = m
            if m1.distance < ratio_test * m2.distance:
                good.append(m1)

    if len(good) < 4:
        return np.empty((0, 2), dtype=np.float32), np.empty((0, 2), dtype=np.float32)

    pts1 = np.float32([keypoints1[m.queryIdx].pt for m in good])
    pts2 = np.float32([keypoints2[m.trainIdx].pt for m in good])
    return pts1, pts2


def estimate_homography(
    pts_src: np.ndarray,
    pts_dst: np.ndarray,
    ransac_reproj_threshold: float = 4.0,
) -> Tuple[bool, np.ndarray]:
    """
    Estimate planar homography H mapping src -> dst.
    Returns success flag and H (3x3).
    """
    if pts_src.shape[0] < 4 or pts_dst.shape[0] < 4:
        return False, np.eye(3, dtype=np.float32)
    H, mask = cv.findHomography(
        pts_src, pts_dst, cv.RANSAC, ransac_reproj_threshold, maxIters=5000, confidence=0.999
    )
    if H is None:
        return False, np.eye(3, dtype=np.float32)
    return True, H.astype(np.float32)


def warp_corners(size: Tuple[int, int], H: np.ndarray) -> np.ndarray:
    """
    Warp image corners with homography H.
    size = (h, w)
    Returns (4,2) array of warped corner coordinates.
    """
    h, w = size
    corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
    warped = cv.perspectiveTransform(corners, H).reshape(-1, 2)
    return warped


def compute_canvas_transform(
    corners_list: List[np.ndarray],
) -> Tuple[np.ndarray, Tuple[int, int]]:
    """
    Compute translation matrix T so that all warped corners have non-negative coordinates
    and calculate canvas size.
    corners_list: list of (4,2) arrays for all images warped into panorama space.
    Returns translation matrix T (3x3) and canvas size (height, width).
    """
    all_pts = np.vstack(corners_list)
    min_xy = np.floor(all_pts.min(axis=0)).astype(np.int32)
    max_xy = np.ceil(all_pts.max(axis=0)).astype(np.int32)

    tx = -min(0, int(min_xy[0]))
    ty = -min(0, int(min_xy[1]))

    width = int(max_xy[0] + tx)
    height = int(max_xy[1] + ty)

    T = np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]], dtype=np.float32)
    return T, (height, width)


def feather_blend(
    base: np.ndarray,
    base_mask: np.ndarray,
    overlay: np.ndarray,
    overlay_mask: np.ndarray
) -> np.ndarray:
    """
    Feather blend overlay onto the base using distance-transform-based weights.
    base: float32 images in [0,1].
    overlay: float32 images in [0,1].
    masks: uint8 {0,1}, same HxW.
    """
    # Where no overlap, just add
    only_base = (base_mask == 1) & (overlay_mask == 0)
    only_overlay = (overlay_mask == 1) & (base_mask == 0)
    overlap = (base_mask == 1) & (overlay_mask == 1)

    out = np.zeros_like(base, dtype=np.float32)
    out[only_base] = base[only_base]
    out[only_overlay] = overlay[only_overlay]

    if np.any(overlap):
        # Distance to the edge inside the mask
        # Inverts masks to get background = 1, foreground = 0 for distanceTransform
        inv_base = (1 - base_mask).astype(np.uint8)
        inv_overlay = (1 - overlay_mask).astype(np.uint8)

        dist_base = cv.distanceTransform(inv_base, cv.DIST_L2, 3)
        dist_overlay = cv.distanceTransform(inv_overlay, cv.DIST_L2, 3)

        # Normalize distances in overlap to avoid extreme weights
        eps = 1e-6
        w_base = dist_base / (dist_base + dist_overlay + eps)
        w_overlay = 1.0 - w_base

        # Broadcast weights to channels
        if base.ndim == 3:
            w_base_c = np.repeat(w_base[:, :, None], base.shape[2], axis=2)
            w_overlay_c = 1.0 - w_base_c
        else:
            w_base_c = w_base
            w_overlay_c = w_overlay

        out[overlap] = w_base_c[overlap] * base[overlap] + w_overlay_c[overlap] * overlay[overlap]

    return out


def stitch_images_planar(images: List[np.ndarray]) -> np.ndarray:
    """
    Incremental planar panorama stitching with homography and feather blending.
    images: list of BGR images
    Returns stitched BGR image.
    """
    if len(images) == 0:
        raise ValueError("No images to stitch.")
    if len(images) == 1:
        return images[0]

    # Convert to grayscale at once
    grays = [cv.cvtColor(im, cv.COLOR_BGR2GRAY) for im in images]

    # Homography mapping each image into the first image's coordinate system (panorama space)
    H_to_ref = [np.eye(3, dtype=np.float32)]
    for i in range(1, len(images)):
        # Match current image i to previous image i-1 (good for adjacent 30-50% overlaps)
        pts_prev, pts_cur = detect_and_match(grays[i - 1], grays[i])

        ok, H_cur_to_prev = estimate_homography(pts_cur, pts_prev)
        if not ok:
            # Try to match to the last reliable reference (image 0) as a fallback
            pts_ref, pts_cur2 = detect_and_match(grays[0], grays[i])
            ok2, H_cur_to_ref_direct = estimate_homography(pts_cur2, pts_ref)
            if not ok2:
                raise RuntimeError(f"Failed to estimate homography for image index {i}.")
            H_to_ref.append(H_cur_to_ref_direct)
        else:
            # Chain with previous mapping
            H_to_ref.append(H_to_ref[i - 1] @ H_cur_to_prev)

    # Compute canvas transform and size
    corners_list = [warp_corners((img.shape[0], img.shape[1]), H) for img, H in zip(images, H_to_ref)]
    T, canvas_hw = compute_canvas_transform(corners_list)

    H_to_canvas = [T @ H for H in H_to_ref]

    Hc, Wc = canvas_hw
    # Prepare accumulators
    pano = np.zeros((Hc, Wc, 3), dtype=np.float32)
    mask_accum = np.zeros((Hc, Wc), dtype=np.uint8)

    for idx, (img, Hc_i) in enumerate(zip(images, H_to_canvas)):
        h, w = img.shape[:2]
        # Warp image and its mask to canvas
        warped = cv.warpPerspective(
            img, Hc_i, (Wc, Hc), flags=cv.INTER_LINEAR, borderMode=cv.BORDER_CONSTANT, borderValue=(0, 0, 0)
        )
        mask = np.zeros((h, w), dtype=np.uint8)
        mask[:] = 1
        warped_mask = cv.warpPerspective(
            mask, Hc_i, (Wc, Hc), flags=cv.INTER_NEAREST, borderMode=cv.BORDER_CONSTANT, borderValue=(0, 0, 0)
        )

        # Convert to float [0,1] for blending
        warped_f = warped.astype(np.float32) / 255.0
        pano_f = pano

        # Blend current warped onto accumulator
        pano = feather_blend(pano_f, mask_accum, warped_f, warped_mask)
        mask_accum = np.clip(mask_accum | warped_mask, 0, 1).astype(np.uint8)

    pano_uint8 = np.clip(pano * 255.0, 0, 255).astype(np.uint8)
    return pano_uint8


def process_folder(source_dir: str, target_path: str) -> None:
    files = list_image_files(source_dir)
    if not files:
        logger.warning(f"No images found in {source_dir}")
        return

    # Load images in BGR
    images = []
    for f in files:
        img = cv.imread(f, cv.IMREAD_COLOR)
        if img is None:
            logger.warning(f"Could not read image: {f}")
            continue
        images.append(img)

    if not images:
        logger.warning(f"No readable images in {source_dir}")
        return

    # Determine grid layout based on the number of images
    num_images = len(images)
    grid_layout = utils.calculate_grid_layout(num_images)

    pano = stitch_images_grid(images, grid_layout)

    ensure_dir(os.path.dirname(target_path))
    cv.imwrite(target_path, pano)
    logger.info(f"Stitched {source_dir} (layout {grid_layout}) into {target_path}")


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


def get_user_input():
    """Get category from user input."""
    print("\n=== Image Merger Configuration ===")

    return utils.select_category()


def main():
    category = get_user_input()

    source_root = f"data/{category}/source_images/"
    target_root = f"data/{category}/target_images/"

    logger.info(f"Start image merging...")
    logger.info(f"Source: {source_root}")
    logger.info(f"Target: {target_root}")

    ensure_dir(target_root)
    subdirectories = [
        d for d in sorted(os.listdir(source_root))
        if os.path.isdir(os.path.join(source_root, d))
    ]
    if subdirectories:
        for d in subdirectories:
            src_dir = os.path.join(source_root, d)
            target_file = os.path.join(target_root, f"{d}.png")
            process_folder(src_dir, target_file)
    else:
        # If no subfolders, stitch all images in the root and save one panorama
        target_file = os.path.join(target_root, "panorama.png")
        process_folder(source_root, target_file)


if __name__ == "__main__":
    main()
