import cv2 as cv

def main():
    path = "C://Users//Windows//Repositories//Skripsi//Data_NEW//BOHEA//BOHEA_001.jpg"
    image = cv.imread(path)

    rgb_image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    hsv_image = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    lab_image = cv.cvtColor(image, cv.COLOR_BGR2Lab)

    cv.imshow("original", image)
    cv.imshow("rgb", rgb_image)
    cv.imshow("hsv", hsv_image)
    cv.imshow("lab", lab_image)

    key = cv.waitKey()
    if key == ord('q'):
        exit()

main()