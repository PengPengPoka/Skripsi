import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import find_peaks

def getHist(image, return_data = False):
    non_black_mask = np.any(image != [0, 0, 0], axis=2).astype(np.uint8)

    histograms = []
    color = ['r','g','b']
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
    colour_channel = ['r', 'g', 'b']
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


def main():
    image = cv.imread("BOHEA_test1.jpg")
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

    color_histogram = getHist(image, True)
    peaks = getHistogramPeak(color_histogram, height=100, prominence=100)


main()