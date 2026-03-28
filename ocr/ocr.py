import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Any

from paddleocr import PaddleOCR, PaddleOCRVL


logger = logging.getLogger(__name__)


class OCRModel(ABC):
    """Abstract base class for OCR models"""

    @abstractmethod
    def recognize(self, image_path: str) -> List[Dict[str, Any]]:
        """
        Recognize text from an image
        Returns: List of dictionaries containing text and bounding boxes
        """
        pass


class PaddleOCRModel(OCRModel):
    """PaddleOCR implementation"""

    def __init__(self):
        self.ocr = PaddleOCR(
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False
        )

    def recognize(self, image_path: str) -> List[Dict[str, Any]]:
        result = self.ocr.predict(image_path)
        formatted_results = []
        for res in result:
            res_data = None
            if hasattr(res, 'res'):
                res_data = res.res
            elif isinstance(res, dict) and 'res' in res:
                res_data = res['res']
            elif isinstance(res, dict):
                res_data = res
            if res_data is None:
                continue
            rec_texts = res_data.get('rec_texts', [])
            rec_scores = res_data.get('rec_scores', [])
            rec_polys = res_data.get('rec_polys', [])
            for i, text in enumerate(rec_texts):
                if not text:
                    continue
                confidence = float(rec_scores[i]) if i < len(rec_scores) else 1.0
                if confidence < 0.5:
                    continue
                bbox = rec_polys[i].tolist() if i < len(rec_polys) else [[0, 0], [0, 0], [0, 0], [0, 0]]
                formatted_results.append({
                    'text': text,
                    'confidence': confidence,
                    'bbox': [[float(x), float(y)] for x, y in bbox]
                })
        return formatted_results


class PaddleOCRVLModel(OCRModel):
    """PaddleOCR-VL implementation"""

    def __init__(self):
        self.pipeline = PaddleOCRVL()

    def recognize(self, image_path: str) -> List[Dict[str, Any]]:
        """
        Recognize text from an image using PaddleOCR-VL
        """
        try:
            output = self.pipeline.predict(image_path)
            formatted_results = []
            for res in output:
                parsing_res_list = res.get('parsing_res_list', [])
                for item in parsing_res_list:
                    text = getattr(item, 'content', '')
                    bbox = getattr(item, 'bbox', None)
                    if bbox and isinstance(bbox, list) and len(bbox) == 4:
                        x1, y1, x2, y2 = bbox
                        bbox_formatted = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
                    else:
                        bbox_formatted = []
                    formatted_results.append({'text': text, 'bbox': bbox_formatted})
            if not formatted_results:
                formatted_results.append({'text': '', 'bbox': []})
            return formatted_results
        except Exception as e:
            logger.error(f"Error during PaddleOCR-VL recognition: {e}", exc_info=True)
            return [{'text': f'Error: {str(e)}', 'bbox': []}]


class OCRFactory:
    """Factory class to create OCR models"""

    _models = {}

    @classmethod
    def create_model(cls, model_name: str) -> OCRModel:
        """Create or return a cached OCR model instance"""
        model_name = model_name.lower()

        # Return cached instance if exists
        if model_name in cls._models:
            return cls._models[model_name]

        # Create a new instance
        if model_name == 'paddleocr':
            model = PaddleOCRModel()
        elif model_name == 'paddleocr-vl':
            model = PaddleOCRVLModel()
        else:
            raise ValueError(f"Unknown model: {model_name}")

        # Cache the model
        cls._models[model_name] = model
        return model

    @staticmethod
    def get_available_models() -> List[Dict[str, Any]]:
        return [
            {'id': 'paddleocr', 'name': 'PaddleOCR', 'status': 'ready'},
            {'id': 'paddleocr-vl', 'name': 'PaddleOCR-VL', 'status': 'ready'},
        ]
