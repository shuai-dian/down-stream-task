import cv2
import numpy as np
import tensorly as tl
from tensorly.decomposition import tucker
from tensorly import tucker_to_tensor

def load_image(file_path):
    return cv2.imread(file_path)

def save_image(file_path, image):
    cv2.imwrite(file_path, image)

def change_detection_tucker(image1, image2, rank, threshold_multiplier=2):
    # Tucker decomposition
    core1, factors1 = tucker(image1, rank=rank)
    core2, factors2 = tucker(image2, rank=rank)

    # Tucker reconstruction
    reconstructed_data1 = tucker_to_tensor((core1, factors1))
    reconstructed_data2 = tucker_to_tensor((core2, factors2))

    # Compute the absolute difference between the reconstructed images
    difference = np.abs(reconstructed_data1 - reconstructed_data2)

    # Convert the difference to grayscale
    # gray_difference = cv2.cvtColor(difference.astype(np.uint8), cv2.COLOR_BGR2GRAY)

    # Compute the threshold for change detection
    threshold = np.mean(difference) + threshold_multiplier * np.std(difference)

    # Create a binary change map based on the threshold
    _, change_map = cv2.threshold(difference, threshold, 255, cv2.THRESH_BINARY)

    return change_map

# Load your RGB images
image1 = load_image('img/002/org_60fb71ecd432c5c4_1681269902000.jpg')
image2 = load_image('img/002/org_73fcb9ca7f1531c1_1681270444000.jpg')

# Define desired ranks for the decomposition, adjust according to your data
rank = [100, 100, 3]

# Perform change detection
change_map = change_detection_tucker(image1, image2, rank)

# Save the change map as a PNG image
save_image("change_map_002.png", change_map)
