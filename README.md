# pix

## ocr

It reads text from an image.

### Training

The `ocr/training` module provides a workflow for fine-tuning PP-OCRv5 / PP-OCRv6
detection and recognition models on your own bad-case images.  See
[`ocr/training/README.md`](ocr/training/README.md) for the full guide.

Quick start:

```bash
# One-time PaddleX repo setup
python -m ocr.training setup

# Generate pseudo labels, review/correct them, build dataset, then train
python -m ocr.training auto-label --task det --model PP-OCRv6_medium_det
python -m ocr.training prepare     --task det --input-dir ocr/training_data/bad_cases/det --label-file ocr/training_data/pseudo_labels/det/labels.json
python -m ocr.training train       --task det --model PP-OCRv6_medium_det --epochs 100
```

## pm (image merger)

It merges multiple images into one.

## ps (image splitter)

It splits an image into multiple images.

## uc (unit converter)

It converts a unit of measurement in an image into another unit.

## License

This project is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for details.
