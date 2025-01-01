import base64
import re

import cv2
import numpy as np
import requests

from PIL import Image, ImageDraw

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


def find_dominant_text_color(text_region_rgb, bg_color):
    pixels = text_region_rgb.reshape(-1, 3).astype(np.float32)
    distances = np.linalg.norm(pixels - np.array(bg_color), axis=1)
    text_color = tuple(int(c) for c in pixels[distances.argmax()])
    return text_color


def recognize_and_replace(image_path, source_unit, target_unit, output_path):
    data = run_ocr(image=image_path, mode='baidu_basic_with_coordinates')
    image = cv2.imread(image_path)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb_image)
    draw = ImageDraw.Draw(pil_image)
    for item in data['words_result']:
        text = item['words'].strip()
        match = re.match(r"(\d+(\.\d+)?)(\s*)({})".format(source_unit), text)
        if match:
            value = float(match.group(1))
            converted_value = convert_value(value, source_unit, target_unit)
            converted_text = f"{converted_value:.2f} {target_unit}"
            loc = item['location']
            x, y, w, h = loc['left'], loc['top'], loc['width'], loc['height']
            bg_color = pil_image.getpixel((x, y))
            text_region_rgb = rgb_image[y:y + h, x:x + w]
            text_color = find_dominant_text_color(text_region_rgb, bg_color)
            draw.rectangle(xy=(x, y, x + w, y + h), fill=bg_color)
            draw.text(xy=(x, y), text=converted_text, fill=text_color)
    pil_image.save(output_path)


if __name__ == "__main__":
    input_image_path = "source_images/boot.jpeg"
    output_image_path = "target_images/boot.jpeg"
    input_unit = "cm"
    output_unit = "in"
    recognize_and_replace(input_image_path, input_unit, output_unit, output_image_path)
    print(f"Converted image saved to {output_image_path}")
