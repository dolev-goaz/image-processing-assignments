import numpy as np
import cv2

def add_salt_and_pepper_noise(image: cv2.typing.MatLike, filter_ratio=0.1):
    noisy_image = image.copy()
    total_pixels = noisy_image.size
    num_salt = int(total_pixels * filter_ratio * 0.5)
    num_pepper = int(total_pixels * filter_ratio * 0.5)

    # salt noise (white pixels)
    salt_coords = [np.random.randint(0, i-1, num_salt) for i in noisy_image.shape]
    noisy_image[salt_coords[0], salt_coords[1]] = 255

    # pepper noise (black pixels)
    pepper_coords = [np.random.randint(0, i-1, num_pepper) for i in noisy_image.shape]
    noisy_image[pepper_coords[0], pepper_coords[1]] = 0

    return noisy_image