import cv2 as cv
import numpy as np
import os

# split into different types of channel
def ChannelSplitter(image):
    a,b,c = cv.split(image)

    return a,b,c

# create mask to filter black region of image
def getMask(a, b, c, threshold=0):
    # a,b,c are colour channels
    mask = (a > threshold) & (b > threshold) & (c > threshold).astype(np.uint8)

    return mask

def getAverage(colour_channel, mask):
    channel_average = np.sum(colour_channel * mask) / np.sum(mask) if np.sum(mask) > 0 else 0
    # channel_average = np.average(colour_channel)

    return channel_average

def main():
    # path = "C://Users//Windows//Repositories//Skripsi//Data_NEW//BOHEA//BOHEA_001.jpg"
    image = cv.imread("BOHEA_test1.jpg")

    rgb_image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    hsv_image = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    lab_image = cv.cvtColor(image, cv.COLOR_BGR2Lab)

    r,g,b = ChannelSplitter(rgb_image)
    h,s,v = ChannelSplitter(hsv_image)
    l,a,b = ChannelSplitter(lab_image)

    cv.imshow("r", r)
    cv.imshow("g", g)
    cv.imshow("b", b)

    if cv.waitKey() == ord('q'):
        exit()

    rgb_mask = getMask(r,g,b)
    hsv_mask = getMask(h,s,v)
    lab_mask = getMask(l,a,b)

    print("AVERAGE OF EACH RGB CHANNELS")
    r_avg = getAverage(r,rgb_mask)
    print(f"the mean of red channel is {r_avg:.4f}")
    g_avg = getAverage(g, rgb_mask)
    print(f"the mean of green channel is {g_avg:.4f}")
    b_avg = getAverage(b, rgb_mask)
    print(f"the mean of blue channel is {b_avg:.4f}")

main()