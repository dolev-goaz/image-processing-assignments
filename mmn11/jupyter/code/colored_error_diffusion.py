import cv2
import numpy as np

def generate_colors_bgr(amount: int = None, random=True):
    # FORMAT: BGR
    if not random:
        # blue, green, red, white, black
        return np.asarray([[255,0,0], [0,255,0], [0,0,255], [0,0,0], [255,255,255]])
    
    if not amount:
        amount = np.random.randint(5, 10)
    return np.random.randint(0, 256, (amount, 3)) # 3 components, 0-255

def find_closest_color(color: np.ndarray[np.uint8], colors: np.ndarray[np.uint8]):
    distances = np.linalg.norm(colors - color, axis=1)
    closest_index = np.argmin(distances)
    return colors[closest_index]


def colored_error_diffusion(image: cv2.typing.MatLike, colors: np.ndarray[any] = generate_colors_bgr(random=False)):
    h, w = image.shape[:2]
    errors = np.zeros((h, w, 3))
    out = np.zeros((h, w, 3))
    for y in range(h):
        for x in range(w):
            value: np.ndarray = image[y, x] + errors[y, x]
            clr = find_closest_color(value, colors)
            out[y, x] = clr
            diff = value - out[y,x]
            
            # propagate errors
            if y + 1 < h:
                errors[y + 1, x] += 3/8 * diff
                
            if x + 1 < w:
                errors[y, x + 1] += 3/8 * diff
                
            if y + 1 < h and x + 1 < w:
                # For some reason, it looks better when not propagating the entirety of the error
                errors[y + 1, x + 1] += 1/8 * diff
    
    return out
