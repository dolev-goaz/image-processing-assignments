from colored_error_diffusion import colored_error_diffusion
import cv_utils
import cv2
import os

OUTPUT_FOLDER_NAME = "output"

def main():
    print("Start of program")
    img = cv2.imread("Lenna.png", cv2.IMREAD_COLOR)
    color_diffused_img = colored_error_diffusion(img)

    if not os.path.exists(OUTPUT_FOLDER_NAME):
        os.mkdir(OUTPUT_FOLDER_NAME)
    cv2.imwrite(f"{OUTPUT_FOLDER_NAME}/result.png", color_diffused_img)
    
    cv_utils.display_images(titles=["Before", "After"], images=[img, color_diffused_img])

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()