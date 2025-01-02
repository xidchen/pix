import base64
import os
import re

import cv2
import numpy as np
import requests

from PIL import Image, ImageDraw, ImageFont

import cfg


CONVERSION_FACTORS = {
    ('cm', 'in'): 0.393701,
    ('in', 'cm'): 2.54,
    ('m', 'ft'): 3.28084,
    ('ft', 'm'): 0.3048,
}

def convert_value(value, source_unit, target_unit):
    if (source_unit, target_unit) in CONVERSION_FACTORS:
        return value * CONVERSION_FACTORS[(source_unit, target_unit)]
    raise ValueError(f"Conversion from {source_unit} to {target_unit} is not supported.")


def run_ocr(image: str, mode: str) -> {}:
    res = []
    if mode == 'baidu_accurate_with_coordinates':
        res = run_baidu_ocr(image, accurate=True, with_coordinates=True)
    if mode == 'baidu_basic_with_coordinates':
        res = run_baidu_ocr(image, accurate=False, with_coordinates=True)
    if mode == 'baidu_accurate_without_coordinates':
        res = run_baidu_ocr(image, accurate=True, with_coordinates=False)
    if mode == 'baidu_basic_without_coordinates':
        res = run_baidu_ocr(image, accurate=False, with_coordinates=False)
    if mode == 'baidu_table':
        res = run_baidu_ocr(image, table=True)
    return res


def run_baidu_ocr(image: str,
                  with_coordinates: bool = True,
                  accurate: bool = True,
                  table: bool = False) -> {}:
    url = cfg.baidu_oauth_url
    params = {'grant_type': 'client_credentials',
              'client_id': cfg.baidu_ocr_client_id,
              'client_secret': cfg.baidu_ocr_client_secret}
    response = requests.get(url, params=params)
    access_token = response.json()['access_token']
    params = {'access_token': access_token}
    f = open(image, 'rb')
    im = base64.b64encode(f.read())
    data = {'image': im}
    if table:
        url = cfg.baidu_ocr_table_url
    else:
        if accurate:
            url = cfg.baidu_ocr_accurate_url if with_coordinates else cfg.baidu_ocr_accurate_basic_url
        else:
            url = cfg.baidu_ocr_general_url if with_coordinates else cfg.baidu_ocr_general_basic_url
    headers = {'content-type': 'application/x-www-form-urlencoded'}
    response = requests.post(url, data=data, params=params, headers=headers)
    return response.json()


def get_font_path():
    font_path = ""
    if os.name == "nt":
        font_path = cfg.nt_font_path
    if os.name == "posix":
        font_path = cfg.posix_font_path
    if not os.path.isfile(font_path):
        raise FileNotFoundError(f"No font file found at {font_path}")
    return font_path


def find_dominant_text_color(text_region_rgb, bg_color):
    pixels = text_region_rgb.reshape(-1, 3).astype(np.float32)
    distances = np.linalg.norm(pixels - np.array(bg_color), axis=1)
    text_color = tuple(int(c) for c in pixels[distances.argmax()])
    return text_color


def resize_text_to_fit(draw, text, font_path, max_width, max_height):
    font_size = 1
    font = ImageFont.truetype(font_path, font_size)
    text_bbox = draw.textbbox((0, 0), text, font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]
    while text_width <= max_width and text_height <= max_height:
        font_size += 1
        font = ImageFont.truetype(font_path, font_size)
        text_bbox = draw.textbbox((0, 0), text, font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
    font = ImageFont.truetype(font_path, font_size - 1)
    return font


def recognize_and_replace(image_path, source_unit, target_unit, output_path):
    data = run_ocr(image_path, 'baidu_basic_with_coordinates')
    image = cv2.imread(image_path)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb_image)
    draw = ImageDraw.Draw(pil_image)
    font_path = get_font_path()
    for item in data['words_result']:
        text = item['words'].strip().lower()
        match = re.match(rf"(\d+(\.\d+)?)(\s*)({source_unit})", text)
        if match:
            value = float(match.group(1))
            converted_value = convert_value(value, source_unit, target_unit)
            converted_text = f"{converted_value:.2f} {target_unit}"
            loc = item['location']
            x, y, w, h = loc['left'], loc['top'], loc['width'], loc['height']
            bg_color = pil_image.getpixel((x + w - 1, y))
            text_region_rgb = rgb_image[y:y + h, x:x + w]
            text_color = find_dominant_text_color(text_region_rgb, bg_color)
            draw.rectangle((x, y, x + w, y + h), bg_color)
            font = resize_text_to_fit(draw, converted_text, font_path, w, h)
            text_bbox = draw.textbbox((0, 0), converted_text, font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
            text_x = x + (w - text_width) // 2
            text_y = y + (h - text_height) // 2
            draw.text((text_x, text_y), converted_text, text_color, font)
    pil_image.save(output_path)


if __name__ == "__main__":
    try:
        os.makedirs(cfg.source_dir, exist_ok=True)
        os.makedirs(cfg.target_dir, exist_ok=True)
        input_image_name = input("Enter the path of the image to process: ").strip()
        input_image_path = os.path.join(cfg.source_dir, os.path.basename(input_image_name))
        if not os.path.isfile(input_image_path):
            raise FileNotFoundError(f"No file found at {input_image_path}")
        output_image_name = os.path.basename(input_image_path)
        output_image_path = os.path.join(cfg.target_dir, output_image_name)
        input_unit = "cm"
        output_unit = "in"
        recognize_and_replace(input_image_path, input_unit, output_unit, output_image_path)
        print(f"Converted image saved to {output_image_path}")
    except Exception as e:
        print(f"An error occurred: {e}")
