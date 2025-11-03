import cv2 as cv
import pandas as pd
import numpy as np
from pathlib import Path
from skimage.feature import graycomatrix, graycoprops

def getGrayCoFeatures(image: np.ndarray, distance: list, angles: list, levels = 256) -> np.array:
    glcm = graycomatrix(image, distance, angles, levels)
    
    contrast = graycoprops(glcm, 'contrast')
    dissimilarity = graycoprops(glcm, 'dissimilarity')
    homogeneity = graycoprops(glcm, 'homogeneity')
    energy = graycoprops(glcm, 'energy')
    correlation = graycoprops(glcm, 'correlation')

    glcm_features = np.hstack([contrast, dissimilarity, homogeneity, energy, correlation])
    
    return glcm_features

def Cropper(src: np.ndarray):
    gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
    ret, binary = cv.threshold(gray, 1, 255, cv.THRESH_BINARY)
    
    contours, _ = cv.findContours(binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE) 
    if contours:
        main_contour = max(contours, key=cv.contourArea)
        x, y, w, h = cv.boundingRect(main_contour)
        cropped_image = src[y:y+h, x:x+w]
        
    return cropped_image
    
def main():
    DATA_DIR = Path('Testing Results').resolve()
    
    filename = 'PF_RGB_085.tif'
    file_path = DATA_DIR / 'Score 4' / filename
    
    image = cv.imread(file_path)
    cropped = Cropper(image)
    
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    
    features = getGrayCoFeatures(gray, [5], [0, np.pi/2])
    # print(features.shape)
    # print(features)
    
    df = pd.DataFrame(features,
                      columns=['contrast_0', 'contrast_90',
                               'dissimilarity_0', 'dissimilarity_90',
                               'homogeneity_0', 'homogeneity_90',
                               'energy_0', 'energy_90',
                               'correlation_0', 'correlation_90'])
    print(df.head())
    
    df.to_csv(f'GLCM_{filename}_cropped.csv', index=False)
    
    
    cv.imshow('image', cropped)
    cv.waitKey()
    cv.destroyAllWindows()
    
if __name__ == "__main__":
    main()