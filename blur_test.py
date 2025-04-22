from fileinput import filename

import cv2 as cv
import os
from matplotlib import pyplot as plt

def main():
    i = 1
    while i < 6:
        directory = os.getcwd()
        # img_path = directory + "//Data_NEW//BOHEA//BOHEA_005.jpg"
        filename = 'BOHEA_test'+ str(i) + '.jpg'
        img = cv.imread(filename)
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

        # check if image exist
        if img is None:
            print("No images read")
            exit()

        # testing blurring methods
        mean_blur = cv.blur(img, (5,5))
        gaussian_blur = cv.GaussianBlur(img, (5,5), 0)
        median_blur = cv.medianBlur(img, 5)
        bilateral_filter = cv.bilateralFilter(img, 9, 75, 75)

        # create 2x3 grid for the plots
        fig, axes = plt.subplots(2,3, figsize=(15, 10))

        # show original image
        axes[0, 0].imshow(img)
        axes[0, 0].set_title('original image')
        axes[0, 0].axis('off')

        # show mean blurred image
        axes[0, 1].imshow(mean_blur)
        axes[0, 1].set_title('mean blurred image')
        axes[0, 1].axis('off')

        # show gaussian blurred image
        axes[0, 2].imshow(gaussian_blur)
        axes[0, 2].set_title('gaussian blurred image')
        axes[0, 2].axis('off')

        # show median blurred image
        axes[1, 0].imshow(median_blur)
        axes[1, 0].set_title('median blurred image')
        axes[1, 0].axis('off')

        # show bilateral filtered image
        axes[1, 1].imshow(bilateral_filter)
        axes[1, 1].set_title('bilateral filtered image')
        axes[1, 1].axis('off')

        axes[1, 2].axis('off') # hide empty subplot

        plt.tight_layout()
        plt.show()

        i += 1

main()