"""
Application of 5x5 median blur to original image
Conversion of original image into RGB, HSV, and LAB color space
Cropping the image with a circular mask via edge detection and cup mask detection
Dynamic indexing of images to search files that ends with .jpg
"""

import cv2 as cv
import numpy as np
import re
import time
from pathlib import Path

def Preprocessing(src_path,
                  radius,
                  variant,
                  color_mode,
                  output_dst,
                  output_extention='.tif'):
    image_paths = sorted(list(src_path.glob("*.jpg")))
    failed_images = []

    if not image_paths:
        print(f"No images found in '{src_path}'")
        return failed_images

    print(f"Processing [{len(image_paths)}] in {src_path.name} directory")

    for image_path in image_paths:
        image = cv.imread(str(image_path))
        print(f"'{image_path.name}' image loaded")
        # print(image.shape)
        
        image_id = getImageID(image_path)
        
        if image is not None:
            blurred_image = cv.medianBlur(image, 5)
            mask = getMask(image, radius)
            
            if color_mode == 'RGB':
                blurred_image = cv.bitwise_and(blurred_image, blurred_image, mask=mask)
            
            elif color_mode == 'HSV':
                blurred_image = cv.cvtColor(blurred_image, cv.COLOR_BGR2HSV)
                
                
            elif color_mode == 'LAB':
                blurred_image = cv.cvtColor(blurred_image, cv.COLOR_BGR2Lab)
            
            blurred_image = cv.bitwise_and(blurred_image, blurred_image, mask=mask)
            
            filename = output_dst / f"{variant}_{color_mode}_{''.join(image_id)}{output_extention}"
            cv.imwrite(filename, blurred_image)
            
            print(f"--- Saving '{filename.name}' ---")
            
        else:
            print(f"--- Failed to read image at '{image_path}' ---")
            failed_images.append(image_path)
            continue
        
    return failed_images

def adaptiveCanny(src, sigma=0.33):
    src = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
    median = np.median(src)
    lower = max(0, (1 - sigma * median))
    upper = min(255, (1 + sigma) * median)
    edge = cv.Canny(src, lower, upper)
    
    return edge

def getImageID(image_path):
    file_id = re.findall(r'\d+', image_path.stem)
    
    return file_id

def getMask(src, radius, k_size=11):
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (k_size,k_size))
    
    edge = adaptiveCanny(src=src, sigma=0.75)
    edge = cv.morphologyEx(edge, cv.MORPH_CLOSE, kernel, iterations=1)
    
    circle_coor = getCircles(edge, src)
    mask = np.zeros(shape=src.shape[:2], dtype=np.uint8)
    cv.circle(mask, (circle_coor[:2]), radius, (255,255,255), cv.FILLED)
    
    return mask
    
def getCircles(edge_src, original_src, param1 = 100, param2 = 50, minRadius = 0, maxRadius = 0):
    center_coor = np.array([])
    circles = cv.HoughCircles(edge_src, cv.HOUGH_GRADIENT, 1, 200,
                              param1=param1, param2=param2, minRadius=minRadius, maxRadius=maxRadius)
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0,:]:
            center_coor = np.array([i[0], i[1], i[2]])
            
    else:
        hsv = cv.cvtColor(original_src, cv.COLOR_BGR2HSV)
        h, s, v = cv.split(hsv)
        cup_mask = (v >= 220) & (s <= 60)
        tea_mask = (~cup_mask).astype('uint8') * 255
        
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (7,7))
        tea_mask = cv.morphologyEx(tea_mask, cv.MORPH_CLOSE, kernel, iterations=1)
        
        contours, hierarchy = cv.findContours(tea_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        
        contour_area = max(contours, key=cv.contourArea)
        
        (x, y), r = cv.minEnclosingCircle(contour_area)
        center_coor = np.array([int(x), int(y), int(r)])   

    return center_coor

def main():
    BASE_DIR = Path(__file__).absolute().parent
    DATA_DIR = BASE_DIR / 'Data_NEW'
    
    OUTPUT_DIR = BASE_DIR / 'Preprocessed Images NEW'
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    TEA_VARIANTS = ['BOHEA', 'BOP', 'BOPF', 'DUST', 'DUST_II', 'F_I', 'F_II', 'PF', 'PF_II', 'PF_III']
    COLOR_MODE = ['RGB', 'HSV', 'LAB']
    
    process_start = time.time()
    
    for variant in TEA_VARIANTS:
        variant_output_dir = OUTPUT_DIR / variant
        variant_output_dir.mkdir(exist_ok=True)
        
        image_dir = DATA_DIR / variant
        
        for mode in COLOR_MODE:
            mode_output_dir = variant_output_dir / f"{variant}_{mode}"
            mode_output_dir.mkdir(exist_ok=True)
            
            Preprocessing(image_dir,
                          100,
                          variant,
                          mode,
                          mode_output_dir)
    
    process_end = time.time()
    processing_time = process_end - process_start
    print(f"Processing time: {processing_time:2.0f}")
    
if __name__ == "__main__":
    main()