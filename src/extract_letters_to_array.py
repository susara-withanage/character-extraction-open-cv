import cv2

from src.is_letter import is_letter
import cv2


def extract_letters_from_image(image, binary_image):
    letters = []

    # Find contours in the binary image
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Sort contours by the y-coordinate (top-to-bottom)
    contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[1])

    # Group contours into rows based on their y-coordinate
    rows = []
    current_row = []
    y_threshold = 10  # Threshold to consider contours in the same row
    previous_y = None

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)

        # If we're starting a new row
        if previous_y is None or (y - previous_y > y_threshold):
            if current_row:
                # Sort the current row by x-coordinate (left-to-right)
                current_row = sorted(current_row, key=lambda c: cv2.boundingRect(c)[0])
                rows.append(current_row)
            current_row = []

        current_row.append(contour)
        previous_y = y

    # Add the last row
    if current_row:
        current_row = sorted(current_row, key=lambda c: cv2.boundingRect(c)[0])
        rows.append(current_row)

    # Extract the letters from each row, now in proper string order
    for row in rows:
        for contour in row:
            if is_letter(contour):
                # Get the bounding box of the contour
                x, y, w, h = cv2.boundingRect(contour)

                # Extract the region of interest (ROI) containing the letter
                letter_roi = image[y:y + h, x:x + w]
                letters.append(letter_roi)
        letters.append("N")

    return letters


def extract_letter_from_template(image, binary_image):
    # Find contours in the binary image
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[1])

    # Iterate through each contour
    for i, contour in enumerate(contours):
        # Get the bounding box of the contour
        x, y, w, h = cv2.boundingRect(contour)

        # Extract the region of interest (ROI) containing the letter
        letter_roi = image[y:y + h, x:x + w]

        return letter_roi
