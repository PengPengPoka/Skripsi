from statistics import median

import cv2 as cv
import os

def main():
    i = 1

    while i < 6:
        filename = 'BOHEA_test' + str(i) +'.jpg'
        img = cv.imread(filename)

        # generate blur images
        mean_blur = cv.blur(img, (5, 5))
        gaussian_blur = cv.GaussianBlur(img, (5, 5), 0)
        median_blur = cv.medianBlur(img, 5)
        bilateral_filter = cv.bilateralFilter(img, 9, 75, 75)

        cv.imwrite('Mean blur_' + filename + '.jpg', mean_blur)
        cv.imwrite('Gaussian blur_' + filename + '.jpg', gaussian_blur)
        cv.imwrite('Median blur_' + filename + '.jpg', median_blur)
        cv.imwrite('Bilateral filter_' + filename + '.jpg', bilateral_filter)



        i += 1

        if i > 5:
            exit()

main()