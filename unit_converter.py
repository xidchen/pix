import base64
import os
import re

import cv2
import numpy as np
import requests

from PIL import Image, ImageDraw, ImageFont
from sklearn.cluster import KMeans
from sklearn.exceptions import ConvergenceWarning

import cfg


CONVERSION_FACTORS = {
    ('cm', 'in'): 0.393701,
    ('g', 'oz'): 0.035274,
    ('ml', 'fl oz'): 0.033814,
}


def extract_values(text, unit):
    v_expr = r"\d+(\.\d+)?"
    m_expr = rf"([*x])?\s*({v_expr})?\s*"
    expr = rf"(.*?)\s*({v_expr})\s*({unit})?\s*{m_expr}({unit})?\s*{m_expr}({unit})"
    match = re.match(f"{expr}", text)
    if match:
        prefix = match.group(1)
        values = [float(match.group(2))]
        if match.group(6):
            values.append(float(match.group(6)))
        if match.group(10):
            values.append(float(match.group(10)))
        separator = match.group(5) or match.group(9)
        return prefix, values, separator
    return None, None, None


def convert_value(values, source_unit, target_unit):
    if (source_unit, target_unit) in CONVERSION_FACTORS:
        factor = CONVERSION_FACTORS[(source_unit, target_unit)]
        converted_values = [value * factor for value in values]
        return converted_values
    elif (target_unit, source_unit) in CONVERSION_FACTORS:
        factor = CONVERSION_FACTORS[(target_unit, source_unit)]
        converted_values = [value / factor for value in values]
        return converted_values
    raise ValueError(
        f"Conversion from {source_unit} to {target_unit} is not supported."
    )


def run_ocr(image: str, mode: str) -> {}:
    if mode == 'baidu_accurate_with_coordinates':
        return run_baidu_ocr(image, accurate=True)
    if mode == 'baidu_basic_with_coordinates':
        return run_baidu_ocr(image, accurate=False)


def run_baidu_ocr(image: str, accurate: bool = True) -> {}:
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
    url = cfg.baidu_ocr_accurate_url if accurate else cfg.baidu_ocr_general_url
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


def find_dominant_color(pixels, n_clusters):
    labels = KMeans(n_clusters).fit(pixels).labels_
    dominant_label = np.argmax(np.bincount(labels))
    dominant_color_pixels = pixels[labels == dominant_label]
    return dominant_color_pixels


def find_dominant_background_color(image_rgb, x, y, w, h):
    pixels = image_rgb[y:y + h, x:x + w].reshape(-1, 3).astype(np.float32)
    n_clusters = 2
    dominant_color_pixels = find_dominant_color(pixels, n_clusters)
    n_clusters = 4
    while n_clusters > 0:
        try:
            dominant_color_pixels = find_dominant_color(dominant_color_pixels, n_clusters)
            break
        except ConvergenceWarning:
            n_clusters -= 1
    avg_color = tuple(map(int, dominant_color_pixels.mean(axis=0)))
    return avg_color


def find_dominant_text_color(text_region_rgb, bg_color):
    pixels = text_region_rgb.reshape(-1, 3).astype(np.float32)
    distances = np.linalg.norm(pixels - np.array(bg_color), axis=1)
    top_10_colors = pixels[np.argsort(distances)[-10:]]
    avg_color = tuple(map(int, top_10_colors.mean(axis=0)))
    return avg_color


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


def recognize_and_replace(image_path, conversion_direction, output_path):
    data = run_ocr(image_path, cfg.baidu_ocr_mode)
    image = cv2.imread(image_path)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb_image)
    draw = ImageDraw.Draw(pil_image)
    font_path = get_font_path()
    for item in data['words_result']:
        text = item['words'].strip().lower()
        for source_unit, target_unit in CONVERSION_FACTORS:
            if conversion_direction == 1:
                pass
            elif conversion_direction == 2:
                source_unit, target_unit = target_unit, source_unit
            prefix, values, separator = extract_values(text, source_unit)
            if values:
                try:
                    converted_values = convert_value(values, source_unit, target_unit)
                    converted_text = prefix
                    converted_value_strs = []
                    for val in converted_values:
                        val_str = f"{val:.1f}"
                        if float(val_str) == 0.0:
                            precision = 1
                            while float(val_str) == 0.0:
                                precision += 1
                                val_str = f"{val:.{precision}f}"
                        converted_value_strs.append(val_str)
                    if len(converted_value_strs) == 1:
                        converted_text += f" {converted_value_strs[0]} {target_unit}"
                    elif len(converted_value_strs) == 2:
                        converted_text += (f" {converted_value_strs[0]} {separator}"
                                           f" {converted_value_strs[1]} {target_unit}")
                    elif len(converted_value_strs) == 3:
                        converted_text += (f" {converted_value_strs[0]} {separator}"
                                           f" {converted_value_strs[1]} {separator}"
                                           f" {converted_value_strs[2]} {target_unit}")
                    loc = item['location']
                    x, y, w, h = loc['left'], loc['top'], loc['width'], loc['height']
                    bg_color = find_dominant_background_color(rgb_image, x, y, w, h)
                    text_region_rgb = rgb_image[y:y + h, x:x + w]
                    text_color = find_dominant_text_color(text_region_rgb, bg_color)
                    draw.rectangle((x, y, x + w, y + h), bg_color)
                    font = resize_text_to_fit(draw, converted_text, font_path, w, h)
                    text_bbox = draw.textbbox((0, 0), converted_text, font)
                    text_width = text_bbox[2] - text_bbox[0]
                    text_height = text_bbox[3] - text_bbox[1]
                    text_x = x if prefix else x + (w - text_width) // 2
                    text_y = y if prefix else y + (h - text_height) // 2
                    draw.text((text_x, text_y), converted_text, text_color, font)
                except ValueError:
                    continue
    pil_image.save(output_path)


if __name__ == "__main__":
    try:
        os.makedirs(cfg.source_dir, exist_ok=True)
        os.makedirs(cfg.target_dir, exist_ok=True)
        input_image_name = input(f"Enter the image name to process in {cfg.source_dir}: ").strip()
        input_image_path = os.path.join(cfg.source_dir, os.path.basename(input_image_name))
        if not os.path.isfile(input_image_path):
            raise FileNotFoundError(f"No file found at {input_image_path}")
        output_image_name = os.path.basename(input_image_path)
        output_image_path = os.path.join(cfg.target_dir, output_image_name)
        while True:
            conversion_choice = input("Enter 1 to convert from metric to US units "
                                      "or 2 to convert from US to metric units: ").strip()
            if conversion_choice in ("1", "2"):
                conversion_choice = int(conversion_choice)
                break
            else:
                print("Invalid choice. Please enter 1 or 2.")
        recognize_and_replace(input_image_path, conversion_choice, output_image_path)
        print(f"Converted image saved to {output_image_path}")
    except Exception as e:
        print(f"An error occurred: {e}")
