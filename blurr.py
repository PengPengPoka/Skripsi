import cv2 as cv
import os

def main():
    directory = os.getcwd()
    path = directory + "\\Data_NEW\\BOHEA\\BOHEA_001.jpg"
    img = cv.imread(path)

    if img is None:
        print('no images read')

    median_blur = cv.medianBlur(img, 5)
    cv.imshow("Original", img)
    cv.imshow("Median Blur 5x5", median_blur)

    key = cv.waitKey()

    if key == ord('q'):
        exit()
    elif key == ord('s'):
        cv.imwrite("Median Blur 5x5.jpg", median_blur)
        print("blurred image saved")

main()