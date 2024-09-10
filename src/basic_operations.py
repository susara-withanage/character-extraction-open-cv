import cv2
from matplotlib import pyplot as plt


def read_image(image_path):
    image = cv2.imread(image_path)
    return image


def show_image(image):
    # Convert the image from BGR to RGB format for matplotlib
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.imshow(image_rgb)
    plt.axis('off')  # Turn off axis
    plt.show()
