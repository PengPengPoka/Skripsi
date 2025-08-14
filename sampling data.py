import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import find_peaks
import pandas as pd
import os

def getHist(image, return_data = False):
    non_black_mask = np.any(image != [0, 0, 0], axis=2).astype(np.uint8)

    histograms = []
    color = ['b','g','r']
    for channel, col in enumerate(color):
        histogram = cv.calcHist([image], [channel], mask=non_black_mask, histSize=[256], ranges=[0,256])
        plt.plot(histogram, color=col)
        plt.xlim(0,256)
        histograms.append(histogram.flatten())

    cv.imshow("Image", image)

    plt.xlabel("RGB values")
    plt.ylabel("Pixel Frequency")
    plt.title("RGB values")
    plt.show()

    if return_data:
        return histograms

def getHistogramPeak(histograms, height=None, distance=10, prominence=None):
    colour_channel = ['b', 'g', 'r']
    results = {}

    for i, histogram in enumerate(histograms):
        # find peaks in histogram data
        peaks, properties = find_peaks(histogram, height=height, distance=distance, prominence=prominence)

        # Get the corresponding histogram values for the peaks
        peak_heights = histogram[peaks]

        # Store results for this channel
        results[colour_channel[i]] = {
            'peak_positions': peaks,
            'peak_heights': peak_heights,
            'properties': properties
        }

        # Plot the peaks
        plt.figure(figsize=(10, 4))
        plt.plot(histogram, color=colour_channel[i].lower()[0])
        plt.plot(peaks, peak_heights, "x", color='cyan')

        # plt.text(50,50, "")

        plt.title(f"{colour_channel[i]} Channel Peaks")
        plt.xlabel("RGB Value")
        plt.ylabel("Frequency")
        plt.show()

    return results

def DrawMask(image=None, height=int, width=int, channel=int, points=(), radius=int):
    cv.circle(image, points, radius, [255,255,255], cv.FILLED)

    return image

def nothing():
    pass

def main():
    img_path = 'BOHEA_test1.jpg'
    img = cv.imread(img_path)

    print("Now reading {}".format(img_path))
    print("How do you want to sample?")
    print("1. area transform")
    print("2. Whole sampling")
    choice = input("sampling method: ")

    if int(choice) == 1:
        print("Sampling with hough transform method chosen")

        # sampling points
        points = [
            (320, 162),  # top
            (320, 235),  # mid
            (320, 300),  # bottom
            (250, 235),  # right
            (390, 235)   # left
        ]
        radius = 20

        cv.circle(img,points[0],radius,[0,0,255],2)
        cv.circle(img, points[1], radius, [0, 0, 255], 2)
        cv.circle(img, points[2], radius, [0, 0, 255], 2)
        cv.circle(img, points[3], radius, [0, 0, 255], 2)
        cv.circle(img, points[4], radius, [0, 0, 255], 2)


        height, width, channel = img.shape
        canvas = np.zeros((height,width,channel), dtype=np.uint8)
        for x,y in points:
            # print(x,y)
            mask = DrawMask(canvas, height, width, channel, (x,y), radius)


        cv.imshow("area sampling", mask)
        key = cv.waitKey()
        if key == ord('q'):
            exit()

    elif int(choice) == 2:
        print("Sampling without hough transform method chosen")
        img_histogram = getHist(img,True)
        peak_data = getHistogramPeak(img_histogram, height = 100, distance = 10, prominence = 100)
        print(f"Red channel peaks at values: {peak_data['r']['peak_positions']}, with height: {peak_data['r']['peak_heights']}")
        print(f"Green channel peaks at values: {peak_data['g']['peak_positions']}, with height: {peak_data['g']['peak_heights']}")
        print(f"Blue channel peaks at values: {peak_data['b']['peak_positions']}, with height: {peak_data['b']['peak_heights']}")


if __name__ == "__main__":
    main()