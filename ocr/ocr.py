import logging
import os
from abc import ABC, abstractmethod
from typing import List, Dict, Any

from paddleocr import PaddleOCR
from PIL import Image


# Disable PaddleX vLLM plugin registration to avoid conflicts
os.environ["VLLM_PLUGINS"] = ""


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


class DeepSeekOCRModel(OCRModel):
    """DeepSeek OCR implementation"""

    def __init__(self):
        try:
            # Import vLLM dependencies inside __init__ to avoid early loading
            from vllm import LLM, SamplingParams
            from vllm.model_executor.models.deepseek_ocr import NGramPerReqLogitsProcessor

            self.LLM = LLM
            self.SamplingParams = SamplingParams
            self.NGramPerReqLogitsProcessor = NGramPerReqLogitsProcessor

            # Create a model instance
            self.llm = LLM(
                model="deepseek-ai/DeepSeek-OCR",
                enable_prefix_caching=False,
                mm_processor_cache_gb=0,
                logits_processors=[NGramPerReqLogitsProcessor]
            )
            logger.info("DeepSeek OCR model initialized successfully")
        except ImportError as e:
            logger.error(f"Failed to import vLLM dependencies: {e}")
            raise ImportError(
                "DeepSeek OCR requires vLLM. Install it with: uv add vllm"
            ) from e
        except Exception as e:
            logger.error(f"Failed to initialize DeepSeek OCR model: {e}")
            raise

    def recognize(self, image_path: str) -> List[Dict[str, Any]]:
        """
        Recognize text from an image using DeepSeek OCR
        """
        try:
            # Load and prepare the image
            image = Image.open(image_path).convert("RGB")

            # Prepare input for the model
            prompt = "<image>\nFree OCR."
            model_input = {
                "prompt": prompt,
                "multi_modal_data": {"image": image}
            }

            # Configure sampling parameters with ngram logit processor args
            sampling_params = self.SamplingParams(
                temperature=0.0,
                max_tokens=8192,
                extra_args=dict(
                    ngram_size=30,
                    window_size=90,
                    whitelist_token_ids=[128821, 128822]  # whitelist: <|img_start|>, <|img_end|>
                ),
                skip_special_tokens=False
            )

            # Generate output
            model_outputs = self.llm.generate(model_input, sampling_params)

            # Parse and format results
            formatted_results = []
            for output in model_outputs:
                text = output.outputs[0].text
                # DeepSeek OCR returns markdown-formatted text
                # Split by newlines to get individual text blocks
                lines = [line.strip() for line in text.split('\n') if line.strip()]

                for i, line in enumerate(lines):
                    formatted_results.append({
                        'text': line,
                        'confidence': 1.0,  # DeepSeek OCR doesn't provide per-line confidence
                        'bbox': [[0, i * 20], [500, i * 20], [500, (i + 1) * 20], [0, (i + 1) * 20]]
                    })

            if not formatted_results:
                formatted_results.append({
                    'text': '',
                    'confidence': 0.0,
                    'bbox': [[0, 0], [100, 0], [100, 20], [0, 20]]
                })

            return formatted_results

        except Exception as e:
            logger.error(f"Error during DeepSeek OCR recognition: {e}", exc_info=True)
            return [{
                'text': f'Error: {str(e)}',
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
        else:
            raise ValueError(f"Unknown model: {model_name}")

        # Cache the model
        cls._models[model_name] = model
        return model

    @staticmethod
    def get_available_models() -> List[Dict[str, Any]]:
        return [
            {'id': 'paddleocr', 'name': 'PaddleOCR', 'status': 'ready'},
            {'id': 'deepseek', 'name': 'DeepSeek OCR', 'status': 'ready'},
        ]