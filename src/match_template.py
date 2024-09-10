import cv2


def match_with_templates(extracted_letter, templates):
    best_match = None
    best_match_score = -1
    best_template_letter = None

    # Ensure the extracted letter is grayscale and in the same format as the template (CV_8U)
    extracted_letter_resized = cv2.resize(extracted_letter, (28, 28))  # Resize if needed
    extracted_letter_resized = cv2.cvtColor(extracted_letter_resized, cv2.COLOR_BGR2GRAY)  # Ensure grayscale if needed
    extracted_letter_resized = extracted_letter_resized.astype('uint8')  # Convert to 8-bit if not already

    # Compare with the templates
    for letter, template_image in templates.items():
        # Resize template image to match extracted letter size
        template_image_resized = cv2.resize(template_image, (28, 28))

        # Convert template image to grayscale and the same type
        template_image_resized = template_image_resized.astype('uint8')  # Ensure 8-bit
        if template_image_resized.ndim == 3:
            template_image_resized = cv2.cvtColor(template_image_resized, cv2.COLOR_BGR2GRAY)  # Ensure grayscale

        # Apply template matching
        result = cv2.matchTemplate(extracted_letter_resized, template_image_resized, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(result)

        if max_val > best_match_score:
            best_match_score = max_val
            best_match = template_image
            best_template_letter = letter

    return best_template_letter


