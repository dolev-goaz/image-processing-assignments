import cv2
import numpy as np

def display_images(images: list[cv2.typing.MatLike], titles: list[str], window_name="image display"):
    if len(images) != len(titles):
        raise ValueError("mismatch between images and titles")

    title_height = 30
    padding = 10 # between images

    # convert grayscale to color
    processed_images = [
        cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if len(img.shape) == 2 else img
        for img in images
    ]

    # all images should be the same height
    target_height = max(image.shape[0] for image in processed_images)
    resized_images = [
        cv2.resize(img, (int(img.shape[1] * target_height / img.shape[0]), target_height))
        for img in processed_images
    ]

    # Calculate total width and create a canvas
    total_width = sum(img.shape[1] for img in resized_images) + padding * (len(images) - 1)
    canvas_height = target_height + title_height
    canvas = np.ones((canvas_height, total_width, 3), dtype=np.uint8) * 255 # white background

    current_x = 0
    for img, title in zip(resized_images, titles):
        # image
        canvas[title_height:canvas_height, current_x:current_x + img.shape[1]] = img

        # title
        text_size = cv2.getTextSize(title, cv2.FONT_HERSHEY_COMPLEX, 0.6, 1)[0]
        text_x = current_x + (img.shape[1] - text_size[0]) // 2
        text_y = (title_height + text_size[1]) // 2
        cv2.putText(canvas, title, (text_x, text_y), cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 0, 0), 1) # black

        current_x += img.shape[1] + padding

    # Display the final canvas
    cv2.imshow(window_name, canvas)
