import cv2


def is_letter(contour):
    x, y, w, h = cv2.boundingRect(contour)

    # Calculate aspect ratio
    aspect_ratio = w / float(h)

    # Calculate compactness
    area_contour = cv2.contourArea(contour)
    bounding_rect_area = w * h
    compactness = area_contour / float(bounding_rect_area)

    # Add size constraints (based on font size, adjust min and max as needed)
    min_w, max_w = 5, 100  # Min and max width for letters
    min_h, max_h = 5, 100  # Min and max height for letters

    # Define thresholds for letters based on aspect ratio, compactness, and size constraints
    if min_w < w < max_w and min_h < h < max_h and 0.1 < aspect_ratio < 5 and compactness > 0.2:
        # This contour likely represents a letter
        return True
    else:
        return False
