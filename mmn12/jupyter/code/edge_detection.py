import cv2
import numpy as np

def edge_detection(image: cv2.typing.MatLike):
    kernel = np.array([
        [-1, -2, -1],
        [0,  0,  0],
        [1,  2,  1]
    ])
    
    # apply the filter
    filtered_image = cv2.filter2D(image, -1, kernel)
    
    min_val, max_val = abs(np.min(filtered_image)), np.max(filtered_image)
    
    # normalize 0-1
    filtered_image = (filtered_image + min_val) / (max_val + min_val)
    
    # normalize 0-255
    filtered_image = np.uint8(255 * filtered_image)
    
    return filtered_image