import cv2 as cv
import numpy as np
import pandas as pd
import os

def Process(src, mask):
    blurred_image = cv.medianBlur(src, 5)
    processed_image = cv.bitwise_and(blurred_image, mask)
    
    return processed_image

def main():
    directory = os.getcwd()
    mask_path = os.path.join(directory, f"Testing Results\\mask.jpg")
    image_path = os.path.join(directory, f"Data_NEW\\BOHEA\\BOHEA_001.jpg")
    
    mask = cv.imread(mask_path)
    image = cv.imread(image_path)
    
    if image is None:
        print("No image is read")
    elif mask is None:
        print("no mask is read")
        
    hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    processed_hsv = Process(hsv, mask)
    cv.imwrite("HSV image.tif", processed_hsv)
    
    hsv_tif = cv.imread("HSV image.tif")
    
    flattened_hsv = processed_hsv.reshape(-1,3)
    flattened_hsv_tif = hsv_tif.reshape(-1,3)
    
    df = pd.DataFrame(flattened_hsv, columns = ['H', 'S', 'V'])
    df_tif = pd.DataFrame(flattened_hsv_tif, columns = ['H', 'S', 'V'])
    
    df.to_csv('HSV Read Test.csv', index=False)
    df_tif.to_csv('HSV Tif Read Test.csv', index=False)
    

main()