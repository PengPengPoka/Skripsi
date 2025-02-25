import cv2 as cv
import numpy as np
import os
import csv

# buat trackbar
def on_trackbar():
    pass

# to crop the image circularly
def crop_trackbar(height, width, px, py, radius, src):
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    img = src.copy()

    cv.circle(canvas, (px,py), radius, [255,255,255], cv.FILLED)

    cropped = cv.bitwise_and(img, canvas,mask=None)

    # cv.imshow("canvas", canvas)
    # cv.imshow("crop", cropped)
    return cropped

def main():
    # directory location
    home = os.getcwd()
    # variants = ["DUST", "DUST2", "DUST3", "BOHEA", "BOP", "BOPF", "BOPF1", "F1", "F2", "PF", "PF2", "PF3", "BP"]
    # img_path = home + "\\Data_21-09-2023[1]\\Warna\\" + variants[0] + "_1-1.jpg"
    img_path = home + "//Data_NEW//BOHEA//BOHEA_005.jpg"
    img = cv.imread(img_path) # read image

    if img is None:         # check if image exist
        print(img_path)
        print("no image detected")

    else:
        # get image dimensions
        height = img.shape[0]
        width = img.shape[1]
        channel = img.shape[2]

        # create trackbar
        cv.namedWindow("crop trackbar")
        cv.createTrackbar("x space", "crop trackbar", 0, width - 1, on_trackbar)
        cv.createTrackbar("y space", "crop trackbar", 0, height - 1, on_trackbar)
        cv.createTrackbar("radius", "crop trackbar", 0, 450, on_trackbar)

        cv.setTrackbarPos("x space", "crop trackbar", 320)
        cv.setTrackbarPos("y space", "crop trackbar", 232)
        cv.setTrackbarPos("radius", "crop trackbar", 94)

        while True:
            # get trackbar position
            x = cv.getTrackbarPos("x space", "crop trackbar")
            y = cv.getTrackbarPos("y space", "crop trackbar")
            radius = cv.getTrackbarPos("radius", "crop trackbar")

            # draw circle unto image
            img_copy = img.copy()
            cv.circle(img_copy, (x, y), radius, [0, 0, 255], 2)

            crop_img = crop_trackbar(height, width, x, y, radius, img)      # get cropped image
            cv.imshow("crop trackbar", img_copy)
            cv.imshow("crop", crop_img)
            key = cv.waitKey(3)

            if key == ord('q'):     # break loop
                break
            elif key == ord('s'):   # save radius parameter
                data = [x, y, radius]
                with open("radius_parameter.csv", mode='w', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow(['Parameter'])
                    for item in data:
                        writer.writerow([item])
            elif key == ord('i'):
                filename = input("insert filename: ")
                cv.imwrite((filename + ".jpg"), crop_img)


if __name__ == "__main__":
    main()