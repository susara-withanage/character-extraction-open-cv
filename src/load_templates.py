import os

import cv2

from src.extract_letters_to_array import extract_letter_from_template
from src.preprocess_image import preprocess_image


def load_templates_for_font_style(font_style, template_dir="templates"):
    templates = {}
    font_style_path = os.path.join(template_dir, str(font_style))

    if os.path.isdir(font_style_path):
        for file_name in os.listdir(font_style_path):
            if file_name.endswith('.png'):
                letter = file_name.split('.')[0].upper()  # Extract letter from file name (e.g., 'A' from 'A.jpg')
                template_image = cv2.imread(os.path.join(font_style_path, file_name))
                binary_image = preprocess_image(template_image)
                template_image = extract_letter_from_template(template_image, binary_image)
                if template_image is not None:
                    templates[letter] = template_image

    return templates
