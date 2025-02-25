import cv2 as cv
import numpy as np
import os

def main():
    directory = os.getcwd()
    img_path = directory + "//Data_NEW//BOHEA//BOHEA_002.jpg"
    img = cv.imread(img_path)
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img_gray = cv.GaussianBlur(img_gray, (5,5), 2)

    # hough transform
    circles = cv.HoughCircles(img_gray, cv.HOUGH_GRADIENT,
                              dp=1,
                              minDist=120,            # Minimum distance between detected centers
                              param1=50,              # Higher threshold for the Canny edge detector
                              param2=5,               # Accumulator threshold for circle detection
                              minRadius=20,           # Minimum circle radius
                              maxRadius=50)           # Maximum circle radius

    if circles is not None:
        # convert circle coordinates to int
        circles = np.uint16(np.around(circles))

        # loop detected circle
        for circle in circles[0,:]:
            x,y,r = circle

            # mask for current circle
            mask = np.zeros_like(img_gray)
            # cv.circle(mask, (x,y), r, 255, thickness=-1)

            # visualize circle
            cv.circle(img, (x,y), r, (0,255,0), 2)
    else:
        print("no circles detected")


    cv.imshow("BOHEA 001", img)
    key = cv.waitKey()
    if key == ord('q'):
        exit()

if __name__ == "__main__":
    main()