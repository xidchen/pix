import logging
import tempfile
import time
from pathlib import Path

import uvicorn
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse

from ocr import OCRFactory


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(title="OCR Web App")

ALLOWED_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif', '.webp'}
MAX_FILE_SIZE = 16 * 1024 * 1024  # 16MB


def allowed_file(filename: str) -> bool:
    """Check if file extension is allowed"""
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS


@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the main HTML page"""
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>OCR Web App</title>
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
                max-width: 900px;
                margin: 0 auto;
                background: white;
                border-radius: 20px;
                box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
                padding: 40px;
            }
            h1 {
                text-align: center;
                color: #333;
                margin-bottom: 10px;
                font-size: 2.5em;
            }
            .subtitle {
                text-align: center;
                color: #666;
                margin-bottom: 40px;
                font-size: 1.1em;
            }
            .model-selector {
                margin-bottom: 20px;
            }
            .model-selector label {
                display: block;
                margin-bottom: 10px;
                font-weight: 600;
                color: #333;
            }
            .model-buttons {
                display: flex;
                gap: 10px;
                flex-wrap: wrap;
            }
            .model-btn {
                flex: 1;
                min-width: 120px;
                padding: 12px 20px;
                border: 2px solid #667eea;
                background: white;
                color: #667eea;
                border-radius: 10px;
                cursor: pointer;
                font-size: 1em;
                font-weight: 600;
                transition: all 0.3s ease;
                position: relative;
            }
            .model-btn:hover {
                background: #f0f0f0;
            }
            .model-btn.active {
                background: #667eea;
                color: white;
            }
            .model-btn.not-ready {
                opacity: 0.5;
                cursor: not-allowed;
            }
            .model-btn .status-badge {
                font-size: 0.7em;
                display: block;
                margin-top: 4px;
            }
            .file-upload {
                margin-bottom: 20px;
            }
            .file-upload label {
                display: block;
                margin-bottom: 10px;
                font-weight: 600;
                color: #333;
            }
            .file-input-wrapper {
                position: relative;
                display: inline-block;
                width: 100%;
            }
            .file-input-wrapper input[type="file"] {
                position: absolute;
                left: -9999px;
            }
            .file-input-label {
                display: block;
                padding: 20px;
                background: #f8f9fa;
                border: 2px dashed #667eea;
                border-radius: 10px;
                text-align: center;
                cursor: pointer;
                transition: all 0.3s ease;
            }
            .file-input-label:hover {
                background: #e9ecef;
                border-color: #764ba2;
            }
            .file-input-label.has-file {
                background: #e7f3ff;
                border-color: #667eea;
                border-style: solid;
            }
            .file-input-label.dragging {
                background: #d4e9ff;
                border-color: #764ba2;
                transform: scale(1.02);
            }
            .image-preview {
                margin-top: 20px;
                display: none;
            }
            .image-preview.show {
                display: block;
            }
            .image-preview img {
                max-width: 100%;
                border-radius: 10px;
                box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
            }
            .image-preview-label {
                display: block;
                margin-bottom: 10px;
                font-weight: 600;
                color: #333;
            }
            .submit-btn {
                width: 100%;
                padding: 15px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                border: none;
                border-radius: 10px;
                font-size: 1.1em;
                font-weight: 600;
                cursor: pointer;
                transition: transform 0.2s ease;
            }
            .submit-btn:hover:not(:disabled) {
                transform: translateY(-2px);
            }
            .submit-btn:disabled {
                opacity: 0.6;
                cursor: not-allowed;
            }
            .results-section {
                margin-top: 40px;
                display: none;
            }
            .results-section.show {
                display: block;
            }
            .results-header {
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 20px;
            }
            .results-header h2 {
                color: #333;
            }
            .copy-btn {
                padding: 10px 20px;
                background: #28a745;
                color: white;
                border: none;
                border-radius: 8px;
                cursor: pointer;
                font-weight: 600;
            }
            .copy-btn:hover {
                background: #218838;
            }
            .results-content {
                background: #f8f9fa;
                padding: 20px;
                border-radius: 10px;
                min-height: 150px;
                white-space: pre-wrap;
                word-wrap: break-word;
                line-height: 1.6;
                color: #333;
            }
            .loading {
                text-align: center;
                padding: 40px;
                display: none;
            }
            .loading.show {
                display: block;
            }
            .spinner {
                border: 4px solid #f3f3f3;
                border-top: 4px solid #667eea;
                border-radius: 50%;
                width: 50px;
                height: 50px;
                animation: spin 1s linear infinite;
                margin: 0 auto 20px;
            }
            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }
            .error-message, .success-message {
                padding: 15px;
                border-radius: 10px;
                margin-top: 20px;
                display: none;
            }
            .error-message {
                background: #f8d7da;
                color: #721c24;
            }
            .success-message {
                background: #d4edda;
                color: #155724;
            }
            .error-message.show, .success-message.show {
                display: block;
            }
            .details-section {
                margin-top: 20px;
                padding: 15px;
                background: #fff3cd;
                border-radius: 10px;
                border-left: 4px solid #ffc107;
            }
            .details-section h3 {
                margin-bottom: 10px;
                color: #856404;
            }
            .detail-item {
                margin: 5px 0;
                color: #856404;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🔍 OCR Web App</h1>
            <p class="subtitle">Extract text from images using multiple OCR engines</p>

            <div class="upload-section">
                <div class="model-selector">
                    <label>Select OCR Model:</label>
                    <div class="model-buttons" id="modelButtons">
                        <!-- Models will be loaded dynamically -->
                    </div>
                </div>

                <div class="file-upload">
                    <label>Upload Image:</label>
                    <div class="file-input-wrapper">
                        <input type="file" id="fileInput" accept="image/*">
                        <label for="fileInput" class="file-input-label" id="fileLabel">
                            <span>📁 Click to select an image or drag & drop</span>
                            <br>
                            <small>Supported: PNG, JPG, JPEG, BMP, TIFF, WEBP (Max 16MB)</small>
                        </label>
                    </div>
                </div>

                <div class="image-preview" id="imagePreview">
                    <label class="image-preview-label">Preview:</label>
                    <img id="previewImage" src="" alt="Image preview">
                </div>

                <button class="submit-btn" id="submitBtn" disabled>Perform OCR</button>
            </div>

            <div class="loading" id="loading">
                <div class="spinner"></div>
                <p>Processing image...</p>
            </div>

            <div class="error-message" id="errorMessage"></div>
            <div class="success-message" id="successMessage"></div>

            <div class="results-section" id="resultsSection">
                <div class="results-header">
                    <h2>Extracted Text</h2>
                    <button class="copy-btn" id="copyBtn">📋 Copy Text</button>
                </div>
                <div class="results-content" id="resultsContent"></div>

                <div class="details-section" id="detailsSection">
                    <h3>📊 Details</h3>
                    <div class="detail-item">Model: <strong id="detailModel"></strong></div>
                    <div class="detail-item">Lines detected: <strong id="detailLines"></strong></div>
                    <div class="detail-item">Average confidence: <strong id="detailConfidence"></strong></div>
                    <div class="detail-item">Processing time: <strong id="detailTime"></strong></div>
                </div>
            </div>
        </div>

        <script>
            let selectedModel = 'paddleocr';
            let selectedFile = null;
            let availableModels = [];

            // Load available models
            async function loadModels() {
                try {
                    const response = await fetch('/api/models');
                    const data = await response.json();
                    availableModels = data.models;

                    const container = document.getElementById('modelButtons');
                    container.innerHTML = '';

                    data.models.forEach((model, index) => {
                        const btn = document.createElement('button');
                        btn.className = 'model-btn';
                        btn.dataset.model = model.id;

                        if (model.status !== 'ready') {
                            btn.classList.add('not-ready');
                            btn.disabled = true;
                        }

                        if (index === 0 && model.status === 'ready') {
                            btn.classList.add('active');
                            selectedModel = model.id;
                        }

                        btn.innerHTML = `
                            ${model.name}
                            <span class="status-badge">${model.status === 'ready' ? '✓ Ready' : '⚠ Coming Soon'}</span>
                        `;

                        if (model.status === 'ready') {
                            btn.addEventListener('click', function() {
                                document.querySelectorAll('.model-btn').forEach(b => b.classList.remove('active'));
                                this.classList.add('active');
                                selectedModel = this.dataset.model;
                            });
                        }

                        container.appendChild(btn);
                    });
                } catch (error) {
                    showError('Failed to load models: ' + error.message);
                }
            }

            // File input handling
            const fileInput = document.getElementById('fileInput');
            const fileLabel = document.getElementById('fileLabel');
            const submitBtn = document.getElementById('submitBtn');

            fileInput.addEventListener('change', function(e) {
                selectedFile = e.target.files[0];
                if (selectedFile) {
                    if (selectedFile.size > 16 * 1024 * 1024) {
                        showError('File size exceeds 16MB limit');
                        selectedFile = null;
                        return;
                    }

                    // Check if file is an image
                    if (!selectedFile.type.startsWith('image/')) {
                        showError('Please select a valid image file');
                        selectedFile = null;
                        return;
                    }

                    fileLabel.classList.add('has-file');
                    fileLabel.innerHTML = `
                        <span>✅ ${selectedFile.name}</span>
                        <br>
                        <small>Click to change file</small>
                    `;
                    submitBtn.disabled = false;

                    // Show image preview
                    displayImagePreview(selectedFile);
                }
            });

            // Display image preview
            function displayImagePreview(file) {
                const reader = new FileReader();
                reader.onload = function(e) {
                    const previewImage = document.getElementById('previewImage');
                    const imagePreview = document.getElementById('imagePreview');
                    previewImage.src = e.target.result;
                    imagePreview.classList.add('show');
                };
                reader.readAsDataURL(file);
            }

            // Drag and drop
            fileLabel.addEventListener('dragover', function(e) {
                e.preventDefault();
                e.stopPropagation();
                this.classList.add('dragging');
            });

            fileLabel.addEventListener('dragleave', function(e) {
                e.preventDefault();
                e.stopPropagation();
                this.classList.remove('dragging');
            });

            fileLabel.addEventListener('drop', function(e) {
                e.preventDefault();
                e.stopPropagation();
                this.classList.remove('dragging');

                const files = e.dataTransfer.files;
                if (files.length > 0) {
                    // Manually set the file and trigger validation
                    const file = files[0];
                    
                    // Create a new DataTransfer object to set files
                    const dataTransfer = new DataTransfer();
                    dataTransfer.items.add(file);
                    fileInput.files = dataTransfer.files;
                    
                    fileInput.dispatchEvent(new Event('change'));
                }
            });

            // Submit form
            submitBtn.addEventListener('click', async function() {
                if (!selectedFile) return;

                const loading = document.getElementById('loading');
                const errorMessage = document.getElementById('errorMessage');
                const successMessage = document.getElementById('successMessage');
                const resultsSection = document.getElementById('resultsSection');
                const resultsContent = document.getElementById('resultsContent');

                errorMessage.classList.remove('show');
                successMessage.classList.remove('show');
                resultsSection.classList.remove('show');
                loading.classList.add('show');
                submitBtn.disabled = true;

                try {
                    const formData = new FormData();
                    formData.append('file', selectedFile);
                    formData.append('model', selectedModel);

                    const response = await fetch('/api/ocr', {
                        method: 'POST',
                        body: formData
                    });

                    const data = await response.json();

                    if (data.success) {
                        resultsContent.textContent = data.text || 'No text detected';

                        // Update details
                        document.getElementById('detailModel').textContent = data.model;
                        document.getElementById('detailLines').textContent = data.line_count;
                        document.getElementById('detailConfidence').textContent = 
                            data.avg_confidence ? (data.avg_confidence * 100).toFixed(1) + '%' : 'N/A';
                        document.getElementById('detailTime').textContent = 
                            data.processing_time ? data.processing_time.toFixed(2) + 's' : 'N/A';

                        resultsSection.classList.add('show');
                        successMessage.textContent = `✅ OCR completed successfully using ${data.model}`;
                        successMessage.classList.add('show');
                    } else {
                        errorMessage.textContent = `❌ Error: ${data.error}`;
                        errorMessage.classList.add('show');
                    }
                } catch (error) {
                    errorMessage.textContent = `❌ Error: ${error.message}`;
                    errorMessage.classList.add('show');
                } finally {
                    loading.classList.remove('show');
                    submitBtn.disabled = false;
                }
            });

            // Copy to clipboard
            document.getElementById('copyBtn').addEventListener('click', function() {
                const text = document.getElementById('resultsContent').textContent;
                navigator.clipboard.writeText(text).then(() => {
                    this.textContent = '✅ Copied!';
                    setTimeout(() => {
                        this.textContent = '📋 Copy Text';
                    }, 2000);
                });
            });

            function showError(message) {
                const errorDiv = document.getElementById('errorMessage');
                errorDiv.textContent = message;
                errorDiv.classList.add('show');
                setTimeout(() => errorDiv.classList.remove('show'), 5000);
            }

            // Load models on page load
            window.addEventListener('DOMContentLoaded', loadModels);
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)


@app.get("/api/models")
async def get_models():
    """Get a list of available OCR models"""
    try:
        models = OCRFactory.get_available_models()
        return JSONResponse(content={
            'success': True,
            'models': models
        })
    except Exception as e:
        logger.error(f"Error getting models: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/ocr")
async def perform_ocr(
        file: UploadFile = File(...),
        model: str = Form('paddleocr')
):
    """Perform OCR on an uploaded image"""
    try:
        start_time = time.perf_counter()
        if not file.filename:
            raise HTTPException(status_code=400, detail='No file selected')
        if not allowed_file(file.filename):
            raise HTTPException(
                status_code=400,
                detail=f'File type not allowed. Allowed types: {", ".join(ALLOWED_EXTENSIONS)}'
            )
        available_models = [m['id'] for m in OCRFactory.get_available_models()]
        if model not in available_models:
            raise HTTPException(
                status_code=400,
                detail=f'Invalid model. Available models: {", ".join(available_models)}'
            )
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp_file:
            content = await file.read()
            if len(content) > MAX_FILE_SIZE:
                raise HTTPException(status_code=400, detail='File size exceeds 16MB limit')
            tmp_file.write(content)
            tmp_path = tmp_file.name

        try:
            logger.info(f"Performing OCR with {model} on {file.filename}")
            ocr_model = OCRFactory.create_model(model)
            results = ocr_model.recognize(tmp_path)
            line_count = len(results)
            avg_confidence = sum(r['confidence'] for r in results) / line_count if results else 0
            text = '\n'.join([r['text'] for r in results])
            logger.info(f"OCR completed: {line_count} lines detected, avg confidence: {avg_confidence:.2f}")
            processing_time = time.perf_counter() - start_time
            return JSONResponse(content={
                'success': True,
                'model': model,
                'results': results,
                'text': text,
                'line_count': line_count,
                'avg_confidence': avg_confidence,
                'processing_time': processing_time
            })

        finally:
            Path(tmp_path).unlink(missing_ok=True)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error performing OCR: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return JSONResponse(content={
        'success': True,
        'status': 'healthy'
    })


if __name__ == '__main__':
    logger.info("Starting OCR Web App")
    logger.info("Visit http://localhost:8000 in your browser")
    uvicorn.run(app, host="localhost", port=8000)