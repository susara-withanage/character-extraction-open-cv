import cv2

from src.basic_operations import read_image, show_image
from src.extract_letters_to_array import extract_letters_from_image
from src.load_templates import load_templates_for_font_style
from src.match_template import match_with_templates
from src.preprocess_image import preprocess_image


def main():
    image_path = "images/test_3.png"
    font_family = "1"
    templates = load_templates_for_font_style(font_family)

    image = read_image(image_path)
    binary_image = preprocess_image(image)

    letters = extract_letters_from_image(image, binary_image)
    printed_string = ""

    for letter in letters:
        if isinstance(letter, str):
            # Check if it's a space or a new line
            if letter == "S":
                printed_string += " "  # Add a space
            elif letter == "N":
                printed_string += "\n"  # Add a newline
        else:
            # It's an image, so perform template matching
            printed_string += match_with_templates(letter, templates)

    print(printed_string)


if __name__ == "__main__":
    main()
