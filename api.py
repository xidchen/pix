import base64
import io
import logging
from pathlib import Path
from typing import List

import cv2 as cv
import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from PIL import Image

import utils
from pm import image_merger


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


app = FastAPI(title="Image Merger Web Demo")

# Data directory
DATA_DIR = Path("pm/data")


def image_to_base64(img: np.ndarray) -> str:
    """Convert OpenCV image (BGR) to base64 string."""
    # Convert BGR to RGB
    img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)

    buffer = io.BytesIO()
    pil_img.save(buffer, format="PNG")
    img_str = base64.b64encode(buffer.getvalue()).decode()
    return f"data:image/png;base64,{img_str}"


def get_category_image_files(category: str, subdirectory: str = None) -> List[Path]:
    """
    Get all image files from a category's source_images directory or subdirectory.

    Args:
        category: Category name
        subdirectory: Optional subdirectory name within source_images

    Returns:
        Sorted list of image file paths

    Raises:
        HTTPException: If a category is not found or no images are found
    """
    source_dir = DATA_DIR / category / "source_images"

    if subdirectory:
        source_dir = source_dir / subdirectory

    if not source_dir.exists():
        raise HTTPException(status_code=404, detail=f"Directory not found: {source_dir}")

    # Get all image files (non-recursively if subdirectory specified)
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp')
    image_files = []

    for ext in image_extensions:
        image_files.extend(source_dir.glob(f"*{ext}"))
        image_files.extend(source_dir.glob(f"*{ext.upper()}"))

    if not image_files:
        raise HTTPException(status_code=404, detail=f"No images found in {source_dir}")

    image_files.sort()
    return image_files


def load_images_from_paths(image_paths: List[Path]) -> List[np.ndarray]:
    """
    Load images from file paths using OpenCV.

    Args:
        image_paths: List of image file paths

    Returns:
        List of loaded images as numpy arrays
    """
    images = []
    for img_path in image_paths:
        img = cv.imread(str(img_path), cv.IMREAD_COLOR)
        if img is None:
            logger.warning(f"Could not read image: {img_path}")
            continue
        images.append(img)
    return images


@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the main HTML page."""
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Image Merger</title>
        <style>
            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                padding: 20px;
            }
            .container {
                max-width: 1400px;
                margin: 0 auto;
                background: white;
                border-radius: 20px;
                padding: 40px;
                box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            }
            h1 {
                text-align: center;
                color: #333;
                margin-bottom: 30px;
                font-size: 2.5em;
            }
            .category-section {
                text-align: center;
                margin-bottom: 30px;
            }
            .category-label {
                font-size: 1.3em;
                color: #333;
                margin-bottom: 15px;
                font-weight: 600;
            }
            .category-select {
                width: 100%;
                max-width: 400px;
                padding: 15px;
                font-size: 1.1em;
                border: 2px solid #667eea;
                border-radius: 10px;
                background: white;
                cursor: pointer;
                transition: all 0.3s ease;
            }
            .category-select:hover {
                border-color: #764ba2;
                box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
            }
            .category-select:focus {
                outline: none;
                border-color: #764ba2;
                box-shadow: 0 4px 15px rgba(102, 126, 234, 0.5);
            }
            .load-btn {
                margin-top: 15px;
                padding: 15px 40px;
                font-size: 1.1em;
                border: none;
                border-radius: 10px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                cursor: pointer;
                font-weight: 600;
                transition: all 0.3s ease;
                box-shadow: 0 4px 15px rgba(0,0,0,0.2);
            }
            .load-btn:hover:not(:disabled) {
                transform: translateY(-2px);
                box-shadow: 0 6px 20px rgba(0,0,0,0.3);
            }
            .load-btn:disabled {
                background: #ccc;
                cursor: not-allowed;
                box-shadow: none;
            }
            .image-gallery {
                display: flex;
                flex-wrap: wrap;
                gap: 15px;
                justify-content: center;
                margin-bottom: 30px;
                min-height: 200px;
            }
            .image-item {
                position: relative;
                border: 2px solid #ddd;
                border-radius: 10px;
                overflow: hidden;
                box-shadow: 0 4px 15px rgba(0,0,0,0.1);
                transition: transform 0.3s ease;
            }
            .image-item:hover {
                transform: translateY(-5px);
                box-shadow: 0 8px 25px rgba(0,0,0,0.2);
            }
            .image-item img {
                max-width: 200px;
                max-height: 200px;
                display: block;
            }
            .image-label {
                position: absolute;
                bottom: 0;
                left: 0;
                right: 0;
                background: rgba(0, 0, 0, 0.7);
                color: white;
                padding: 5px;
                font-size: 0.8em;
                text-align: center;
            }
            .button-group {
                display: flex;
                justify-content: center;
                gap: 20px;
                margin-bottom: 30px;
            }
            button {
                padding: 15px 40px;
                font-size: 1.1em;
                border: none;
                border-radius: 10px;
                cursor: pointer;
                font-weight: 600;
                transition: all 0.3s ease;
                box-shadow: 0 4px 15px rgba(0,0,0,0.2);
            }
            .merge-btn {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
            }
            .merge-btn:hover:not(:disabled) {
                transform: translateY(-2px);
                box-shadow: 0 6px 20px rgba(0,0,0,0.3);
            }
            .merge-btn:disabled {
                background: #ccc;
                cursor: not-allowed;
                box-shadow: none;
            }
            .cancel-btn {
                background: #f44336;
                color: white;
            }
            .cancel-btn:hover {
                background: #da190b;
                transform: translateY(-2px);
                box-shadow: 0 6px 20px rgba(0,0,0,0.3);
            }
            .result-section {
                text-align: center;
                margin-top: 30px;
            }
            .result-section h2 {
                color: #333;
                margin-bottom: 20px;
                font-size: 2em;
            }
            .result-image {
                max-width: 100%;
                border: 3px solid #667eea;
                border-radius: 15px;
                box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            }
            .loading {
                display: none;
                text-align: center;
                margin: 20px 0;
            }
            .spinner {
                border: 4px solid #f3f3f3;
                border-top: 4px solid #667eea;
                border-radius: 50%;
                width: 50px;
                height: 50px;
                animation: spin 1s linear infinite;
                margin: 20px auto;
            }
            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }
            .error-message {
                background: #ffebee;
                color: #c62828;
                padding: 15px;
                border-radius: 10px;
                margin: 20px 0;
                display: none;
                border-left: 4px solid #c62828;
            }
            .success-message {
                background: #e8f5e9;
                color: #2e7d32;
                padding: 15px;
                border-radius: 10px;
                margin: 20px 0;
                display: none;
                border-left: 4px solid #2e7d32;
            }
            .info-message {
                background: #e3f2fd;
                color: #1565c0;
                padding: 15px;
                border-radius: 10px;
                margin: 20px 0;
                border-left: 4px solid #1565c0;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🖼️ Image Merger</h1>

            <div class="category-section">
                <div class="category-label">📁 Select Category</div>
                <select id="categorySelect" class="category-select">
                    <option value="">-- Choose a category --</option>
                </select>
                <br>
                <div id="subdirectorySection" style="display: none; margin-top: 15px;">
                    <div class="category-label">📂 Select Subdirectory</div>
                    <select id="subdirectorySelect" class="category-select">
                        <option value="">-- Choose a subdirectory --</option>
                    </select>
                </div>
                <br>
                <button class="load-btn" id="loadBtn" onclick="loadImages()" disabled>
                    📂 Load Images
                </button>
            </div>

            <div class="error-message" id="errorMessage"></div>
            <div class="success-message" id="successMessage"></div>

            <div class="image-gallery" id="imageGallery"></div>

            <div class="button-group" id="buttonGroup" style="display: none;">
                <button class="merge-btn" id="mergeBtn" onclick="mergeImages()">
                    ✨ Merge Images
                </button>
                <button class="cancel-btn" onclick="cancelImages()">
                    ❌ Clear All
                </button>
            </div>

            <div class="loading" id="loading">
                <div class="spinner"></div>
                <p>Merging images... This may take a moment.</p>
            </div>

            <div class="result-section" id="resultSection" style="display: none;">
                <h2>✅ Merged Result</h2>
                <img id="resultImage" class="result-image" src="" alt="Merged Image">
            </div>
        </div>

        <script>
            let currentCategory = '';
            let currentSubdirectory = '';
            let loadedImages = [];

            // Load categories on page load
            window.addEventListener('DOMContentLoaded', async () => {
                await loadCategories();
            });

            async function loadCategories() {
                try {
                    const response = await fetch('/categories');
                    const data = await response.json();

                    const select = document.getElementById('categorySelect');

                    if (data.categories.length === 0) {
                        showError('No categories found in pm/data/');
                        return;
                    }

                    data.categories.forEach(category => {
                        const option = document.createElement('option');
                        option.value = category;
                        option.textContent = category;
                        select.appendChild(option);
                    });

                    select.addEventListener('change', async function() {
                        if (this.value) {
                            await loadSubdirectories(this.value);
                        } else {
                            document.getElementById('subdirectorySection').style.display = 'none';
                            document.getElementById('loadBtn').disabled = true;
                        }
                    });

                } catch (error) {
                    showError('Error loading categories: ' + error.message);
                }
            }

            async function loadSubdirectories(category) {
                try {
                    const response = await fetch(`/categories/${category}/subdirectories`);
                    const data = await response.json();

                    const subdirSelect = document.getElementById('subdirectorySelect');
                    subdirSelect.innerHTML = '<option value="">-- All images (no subdirectory) --</option>';

                    if (data.subdirectories.length > 0) {
                        data.subdirectories.forEach(subdir => {
                            const option = document.createElement('option');
                            option.value = subdir;
                            option.textContent = subdir;
                            subdirSelect.appendChild(option);
                        });
                        
                        document.getElementById('subdirectorySection').style.display = 'block';
                        
                        subdirSelect.addEventListener('change', function() {
                            document.getElementById('loadBtn').disabled = false;
                        });
                        
                        // Enable load button for default "all images" option
                        document.getElementById('loadBtn').disabled = false;
                    } else {
                        // No subdirectories, enable load button immediately
                        document.getElementById('subdirectorySection').style.display = 'none';
                        document.getElementById('loadBtn').disabled = false;
                    }

                } catch (error) {
                    showError('Error loading subdirectories: ' + error.message);
                    document.getElementById('subdirectorySection').style.display = 'none';
                    document.getElementById('loadBtn').disabled = true;
                }
            }

            async function loadImages() {
                const select = document.getElementById('categorySelect');
                const subdirSelect = document.getElementById('subdirectorySelect');
                currentCategory = select.value;
                currentSubdirectory = subdirSelect.value || '';

                if (!currentCategory) {
                    showError('Please select a category');
                    return;
                }

                // Show loading
                document.getElementById('loading').style.display = 'block';
                document.getElementById('resultSection').style.display = 'none';
                document.getElementById('imageGallery').innerHTML = '';
                document.getElementById('buttonGroup').style.display = 'none';
                hideMessages();

                try {
                    const url = currentSubdirectory 
                        ? `/images/${currentCategory}?subdirectory=${encodeURIComponent(currentSubdirectory)}`
                        : `/images/${currentCategory}`;
                    
                    const response = await fetch(url);
                    const data = await response.json();

                    if (!response.ok) {
                        showError(data.detail || 'Failed to load images');
                        return;
                    }

                    if (data.images.length === 0) {
                        const path = currentSubdirectory ? `${currentCategory}/${currentSubdirectory}` : currentCategory;
                        showError(`No images found in: ${path}`);
                        return;
                    }

                    loadedImages = data.images;
                    displayImages(data.images);
                    document.getElementById('buttonGroup').style.display = 'flex';
                    
                    const path = currentSubdirectory ? `${currentCategory}/${currentSubdirectory}` : currentCategory;
                    showSuccess(`Loaded ${data.images.length} images from ${path}`);

                } catch (error) {
                    showError('Error loading images: ' + error.message);
                } finally {
                    document.getElementById('loading').style.display = 'none';
                }
            }

            function displayImages(images) {
                const gallery = document.getElementById('imageGallery');
                gallery.innerHTML = '';

                images.forEach((imgData, index) => {
                    const div = document.createElement('div');
                    div.className = 'image-item';

                    const img = document.createElement('img');
                    img.src = imgData.data;

                    const label = document.createElement('div');
                    label.className = 'image-label';
                    label.textContent = imgData.filename;

                    div.appendChild(img);
                    div.appendChild(label);
                    gallery.appendChild(div);
                });
            }

            async function mergeImages() {
                if (!currentCategory) {
                    showError('No category selected');
                    return;
                }

                if (loadedImages.length < 2) {
                    showError('Need at least 2 images to merge');
                    return;
                }

                // Show loading
                document.getElementById('loading').style.display = 'block';
                document.getElementById('resultSection').style.display = 'none';
                hideMessages();

                try {
                    const response = await fetch('/merge', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json'
                        },
                        body: JSON.stringify({
                            category: currentCategory,
                            subdirectory: currentSubdirectory || null
                        })
                    });

                    const data = await response.json();

                    if (response.ok) {
                        document.getElementById('resultImage').src = data.merged_image;
                        document.getElementById('resultSection').style.display = 'block';
                        
                        const path = currentSubdirectory ? `${currentCategory}/${currentSubdirectory}` : currentCategory;
                        showSuccess(`Images merged successfully from ${path}! Grid layout: [${data.grid_layout}]`);
                    } else {
                        showError(data.detail || 'Failed to merge images');
                    }
                } catch (error) {
                    showError('Error: ' + error.message);
                } finally {
                    document.getElementById('loading').style.display = 'none';
                }
            }

            function cancelImages() {
                loadedImages = [];
                currentCategory = '';
                currentSubdirectory = '';
                document.getElementById('imageGallery').innerHTML = '';
                document.getElementById('resultSection').style.display = 'none';
                document.getElementById('buttonGroup').style.display = 'none';
                document.getElementById('categorySelect').value = '';
                document.getElementById('subdirectorySection').style.display = 'none';
                document.getElementById('loadBtn').disabled = true;
                hideMessages();
            }

            function showError(message) {
                const errorDiv = document.getElementById('errorMessage');
                errorDiv.textContent = message;
                errorDiv.style.display = 'block';
                setTimeout(() => errorDiv.style.display = 'none', 5000);
            }

            function showSuccess(message) {
                const successDiv = document.getElementById('successMessage');
                successDiv.textContent = message;
                successDiv.style.display = 'block';
                setTimeout(() => successDiv.style.display = 'none', 5000);
            }

            function hideMessages() {
                document.getElementById('errorMessage').style.display = 'none';
                document.getElementById('successMessage').style.display = 'none';
            }
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)


@app.get("/categories")
async def get_categories():
    """Get the list of available categories from pm/data/."""
    try:
        categories = utils.discover_categories(str(DATA_DIR))
        return JSONResponse(content={"categories": categories})
    except Exception as e:
        logger.error(f"Error discovering categories: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/categories/{category}/subdirectories")
async def get_subdirectories(category: str):
    """Get all subdirectories within a category's source_images folder."""
    source_dir = DATA_DIR / category / "source_images"

    if not source_dir.exists():
        raise HTTPException(status_code=404, detail=f"Category '{category}' not found")

    try:
        subdirectories = [
            d.name for d in source_dir.iterdir()
            if d.is_dir() and not d.name.startswith('.')
        ]
        subdirectories.sort()

        return JSONResponse(content={
            "category": category,
            "subdirectories": subdirectories
        })
    except Exception as e:
        logger.error(f"Error listing subdirectories for {category}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/images/{category}")
async def get_images(category: str, subdirectory: str = None):
    """Get all source images from a category (optionally from a subdirectory) as base64."""
    image_files = get_category_image_files(category, subdirectory)

    # Load and convert images to base64
    images_data = []
    for img_path in image_files:
        try:
            img = cv.imread(str(img_path), cv.IMREAD_COLOR)
            if img is None:
                logger.warning(f"Could not read image: {img_path}")
                continue

            img_base64 = image_to_base64(img)
            images_data.append({
                "filename": img_path.name,
                "data": img_base64
            })
        except Exception as e:
            logger.warning(f"Error loading image {img_path}: {e}")
            continue

    if not images_data:
        raise HTTPException(status_code=500, detail="Failed to load any images")

    return JSONResponse(content={
        "category": category,
        "subdirectory": subdirectory,
        "count": len(images_data),
        "images": images_data
    })


@app.post("/merge")
async def merge_images(request: dict):
    """Merge images from the specified category and optional subdirectory."""
    category = request.get("category")
    subdirectory = request.get("subdirectory")

    if not category:
        raise HTTPException(status_code=400, detail="Category not specified")

    # Get image files (reusing the helper function)
    image_files = get_category_image_files(category, subdirectory)

    # Load images using OpenCV
    images = load_images_from_paths(image_files)

    if len(images) < 2:
        raise HTTPException(status_code=400, detail="Need at least 2 images to merge")

    if len(images) > 10:
        raise HTTPException(status_code=400, detail="Maximum 10 images allowed")

    merge_path = f"{category}/{subdirectory}" if subdirectory else category
    logger.info(f"Merging {len(images)} images from '{merge_path}'")

    # Calculate grid layout
    num_images = len(images)
    grid_layout = utils.calculate_grid_layout(num_images)

    logger.info(f"Using grid layout: {grid_layout}")

    # Merge images
    try:
        merged_image = image_merger.stitch_images_grid(images, grid_layout)
    except Exception as e:
        logger.error(f"Error during stitching: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to merge images: {str(e)}")

    # Convert to base64
    merged_base64 = image_to_base64(merged_image)

    logger.info(f"Successfully merged {len(images)} images from '{merge_path}'")

    return JSONResponse(content={
        "status": "success",
        "category": category,
        "subdirectory": subdirectory,
        "num_images": len(images),
        "grid_layout": grid_layout,
        "merged_image": merged_base64
    })


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok"}


if __name__ == "__main__":
    logger.info("Starting Image Merger Web Demo")
    logger.info("Visit http://localhost:8000 in your browser")
    uvicorn.run(app, host="0.0.0.0", port=8000)
