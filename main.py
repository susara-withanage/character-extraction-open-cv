import cv2
import os


def load_templates(template_dir):
    templates = {}
    for filename in os.listdir(template_dir):
        if filename.endswith(".jpg"):
            ascii_value = int(os.path.splitext(filename)[0])
            template_image = cv2.imread(os.path.join(template_dir, filename), cv2.IMREAD_GRAYSCALE)
            templates[ascii_value] = template_image
    return templates


def segment_letters(image):
    # Implement letter segmentation here
    return [image]  # Placeholder, replace with actual segmentation code


def match_templates(template_images, letter_regions):
    extracted_text = ""
    for region in letter_regions:
        # Group adjacent regions to form potential letter candidates
        letter_candidates = group_regions(region)
        for candidate in letter_candidates:
            best_match = None
            max_similarity = 0
            region_height, region_width = candidate.shape[:2]
            for ascii_value, template_image in template_images.items():
                template_height, template_width = template_image.shape[:2]
                if region_height < template_height or region_width < template_width:
                    continue  # Skip if the region is smaller than the template

                # Resize the template image to match the size of the region
                resized_template = cv2.resize(template_image, (region_width, region_height))

                # Perform template matching between region and resized template
                result = cv2.matchTemplate(candidate, resized_template, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, _ = cv2.minMaxLoc(result)
                similarity = max_val
                if similarity > max_similarity:
                    max_similarity = similarity
                    best_match = ascii_value
            if best_match is not None:
                extracted_text += chr(best_match)
    return extracted_text


def group_regions(image):
    _, labels, stats, _ = cv2.connectedComponentsWithStats(image, connectivity=4)
    grouped_regions = []
    for i in range(1, stats.shape[0]):  # Exclude background label
        region_label = i
        region_stats = stats[i]
        region_x, region_y, region_w, region_h = region_stats[0:4]
        # Check if the region is not too small
        if region_w > 10 and region_h > 10:  # Adjust threshold as needed
            grouped_regions.append(image[region_y:region_y+region_h, region_x:region_x+region_w])
    return grouped_regions


def main():
    # Load template images
    templates_dir = "templates"
    template_images = load_templates(templates_dir)

    # Load the input image
    input_image = cv2.imread("printed_image.jpg", cv2.IMREAD_GRAYSCALE)

    # Segment the input image into individual regions containing letters
    letter_regions = segment_letters(input_image)

    # Match templates with letter regions
    extracted_text = match_templates(template_images, letter_regions)

    print("Extracted text:", extracted_text)


if __name__ == "__main__":
    main()
