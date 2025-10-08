import logging
import os
import shutil

import cfg


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def copy_files_from_folders():
    """
    Copy files from MVTec AD capsule test folders to the target directory.
    Files are renamed with the pattern: {folder_name}_{file_name}
    """
    source_base_path = cfg.mvtec_raw_image_dir
    source_path = cfg.source_image_dir
    files_per_folder = cfg.source_files_per_folder

    # Remove the target directory if it exists and recreate it
    if os.path.exists(source_path):
        shutil.rmtree(source_path)
        logger.info(f"Removed existing directory: {source_path}")

    # Create a target directory if it doesn't exist
    os.makedirs(source_path, exist_ok=True)

    # Check if a source path exists
    if not os.path.exists(source_base_path):
        raise FileNotFoundError(f"Source path does not exist: {source_base_path}")

    # Get all subdirectories in the source path
    folders = [f for f in os.listdir(source_base_path)
               if os.path.isdir(os.path.join(source_base_path, f))]

    if not folders:
        logger.info(f"No folders found in {source_base_path}")
        return

    total_files_copied = 0

    for folder_name in folders:
        folder_path = os.path.join(source_base_path, folder_name)

        # Get all files in the folder
        files = [f for f in os.listdir(folder_path)
                 if os.path.isfile(os.path.join(folder_path, f))]

        # Sort files to ensure consistent selection
        files.sort()

        # Copy up to files_per_folder files
        files_to_copy = files[:files_per_folder]

        for file_name in files_to_copy:
            source_file = os.path.join(folder_path, file_name)

            # Create a new filename: {folder_name}_{file_name}
            new_file_name = f"{folder_name}_{file_name}"
            target_file = os.path.join(source_path, new_file_name)

            # Copy the file
            shutil.copy2(source_file, target_file)
            logger.info(f"Copied: {folder_name}/{file_name} -> {new_file_name}")
            total_files_copied += 1

    logger.info(f"Total files copied: {total_files_copied}")
    logger.info(f"Files saved to: {source_path}")


if __name__ == "__main__":
    try:
        copy_files_from_folders()
    except Exception as e:
        logger.error(f"An error occurred: {e}")
