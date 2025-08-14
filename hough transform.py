import cv2 as cv
import os
import numpy as np


def nothing():
    pass

def mask(image, points=() , radius=300):
    height, width, _ = image.shape
    canvas = np.zeros((height, width, 1), np.uint8)
    cv.circle(canvas, points, radius, (255,255,255), cv.FILLED)

    return canvas

def crop(image, mask):
    return cv.bitwise_and(image, mask, mask=None)

def main():
    file_path = os.getcwd() + "/Data_NEW/BOHEA/BOHEA_010.jpg"
    image = cv.imread(file_path)

    if image is None:
        print("image does not exist")

    gray = cv.cvtColor(image,cv.COLOR_BGR2GRAY)
    gray = cv.GaussianBlur(gray,(3,3), 0)

    # apply hough transform to grayscale image
    circles = cv.HoughCircles(image = gray,
                              method = cv.HOUGH_GRADIENT,
                              dp = 1,
                              minDist = 20,
                              param1 = 400,
                              param2 = 101,
                              minRadius = 10,
                              maxRadius = 300)

    # get circle data from hough transform
    circles = np.uint16(np.round(circles))
    for circle in circles[0,:]:
        center_x = circle[0]
        center_y = circle[1]
        radius = circle[2]
        masked_image = mask(image, (center_x, center_y), radius)

    # find contours to find extreme points
    contours, _ = cv.findContours(masked_image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    # find extreme points from masked image
    for contour in contours:
        leftmost = tuple(contour[contour[:, :, 0].argmin()][0])
        rightmost = tuple(contour[contour[:, :, 0].argmax()][0])
        topmost = tuple(contour[contour[:, :, 1].argmin()][0])
        bottommost = tuple(contour[contour[:, :, 1].argmax()][0])

    # draw extreme points
    cv.circle(image, leftmost, 5, [0,0,255], -1)
    cv.circle(image, rightmost, 5, [0, 0, 255], -1)
    cv.circle(image, topmost, 5, [0, 0, 255], -1)
    cv.circle(image, bottommost, 5, [0, 0, 255], -1)

    masked_image = cv.cvtColor(masked_image, cv.COLOR_GRAY2BGR)
    cropped = crop(image, masked_image)

    cv.imshow("Hough transform", image)
    cv.imshow("mask image", masked_image)
    cv.imshow("cropped image", cropped)
    key = cv.waitKey()
    if key == ord('q'):
        exit()

main()