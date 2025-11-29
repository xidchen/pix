import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Any

from paddleocr import PaddleOCR


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
            # Access the 'res' dictionary from the result object
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
                if not text:  # Skip empty text
                    continue

                confidence = float(rec_scores[i]) if i < len(rec_scores) else 1.0
                bbox = rec_polys[i].tolist() if i < len(rec_polys) else [[0, 0], [0, 0], [0, 0], [0, 0]]

                formatted_results.append({
                    'text': text,
                    'confidence': confidence,
                    'bbox': [[float(x), float(y)] for x, y in bbox]
                })

        return formatted_results


class DeepSeekOCRModel(OCRModel):
    """DeepSeek OCR implementation"""

    def __init__(self):
        # TODO: Initialize DeepSeek OCR model
        logger.warning("DeepSeek OCR not yet implemented")

    def recognize(self, image_path: str) -> List[Dict[str, Any]]:
        # TODO: Implement DeepSeek OCR recognition
        return [{
            'text': 'DeepSeek OCR - Not yet implemented',
            'confidence': 0.0,
            'bbox': [[0, 0], [100, 0], [100, 20], [0, 20]]
        }]


class DotsOCRModel(OCRModel):
    """DOTS OCR implementation"""

    def __init__(self):
        # TODO: Initialize DOTS OCR model
        logger.warning("DOTS OCR not yet implemented")

    def recognize(self, image_path: str) -> List[Dict[str, Any]]:
        # TODO: Implement DOTS OCR recognition
        return [{
            'text': 'DOTS OCR - Not yet implemented',
            'confidence': 0.0,
            'bbox': [[0, 0], [100, 0], [100, 20], [0, 20]]
        }]


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
        elif model_name == 'deepseek':
            model = DeepSeekOCRModel()
        elif model_name == 'dots':
            model = DotsOCRModel()
        else:
            raise ValueError(f"Unknown model: {model_name}")

        # Cache the model
        cls._models[model_name] = model
        return model

    @staticmethod
    def get_available_models() -> List[Dict[str, Any]]:
        return [
            {'id': 'paddleocr', 'name': 'PaddleOCR', 'status': 'ready'},
            {'id': 'deepseek', 'name': 'DeepSeek OCR', 'status': 'not_implemented'},
            {'id': 'dots', 'name': 'DOTS OCR', 'status': 'not_implemented'}
        ]