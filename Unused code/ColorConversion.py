import cv2 as cv

def cropped(mask, rgb, hsv, lab):
    crop_rgb = cv.bitwise_and(rgb, mask, mask=None)
    crop_hsv = cv.bitwise_and(hsv, mask, mask=None)
    crop_lab = cv.bitwise_and(lab, mask, mask=None)

    return crop_rgb, crop_hsv, crop_lab

def main():
    path = "C:\\Users\\Windows\\Repositories\\Skripsi\\Data_NEW\\BOHEA\\BOHEA_001.jpg"
    # image = cv.imread("C://Users//Windows//Repositories//Skripsi//BOHEA_test1.jpg")
    image = cv.imread(path)
    mask = cv.imread("C:\\Users\\Windows\\Repositories\\Skripsi\\mask.jpg")
    # image = cv.imread(path)

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
    elif key == ord('s'):
        cv.imwrite("hsv_image.jpg", hsv_image)
        cv.imwrite("rgb_image.jpg", rgb_image)
        cv.imwrite("lab_image.jpg", lab_image)
        print("HSV, RGB, and LAB images saved")
    elif key == ord('p'):
        cropped_rgb, cropped_hsv, cropped_lab = cropped(mask, rgb_image, hsv_image, lab_image)
        cv.imwrite("cropped rgb.jpg", cropped_rgb)
        cv.imwrite("cropped hsv.jpg", cropped_hsv)
        cv.imwrite("cropped lab.jpg", cropped_lab)
        print("Cropped HSV, RGB, and LAB images saved")

main()