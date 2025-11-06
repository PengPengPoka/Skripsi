import cv2 as cv
import pandas as pd
import numpy as np
from pathlib import Path
from skimage.feature import graycomatrix, graycoprops
from time import time

def ColorFeatureWriter(data_path: str, mode, output_path: str, output_extention='.csv'):
    csv_files = list(data_path.glob('*.csv'))
    
    if not csv_files:
        print("Directory not found. Skipping")
    
    for csv_file in csv_files:
        data_df = pd.read_csv(csv_file)
        
        color_features = np.concatenate([
            data_df.mean().values,
            data_df.std().values,
            data_df.median().values,
            data_df.mode().iloc[0].values       # make mode shape from (1,3) to (3,)
        ])
        
        color_features = color_features.reshape(1, -1)
        
        if mode == 'RGB':
            columns =  ['B_mean','G_mean','R_mean',
                        'B_std', 'G_std', 'R_std',
                        'B_median', 'G_median', 'R_median',
                        'B_mode', 'G_mode', 'R_mode']
            
        elif mode == 'HSV':
            columns = ['H_mean','S_mean','V_mean',
                       'H_std', 'S_std', 'V_std',
                       'H_median', 'S_median', 'V_median',
                       'H_mode', 'S_mode', 'V_mode']
            
        elif mode == 'LAB':
            columns = ['L_mean','A_mean','B_mean',
                       'L_std', 'A_std', 'B_std',
                       'L_median', 'A_median', 'B_median',
                       'L_mode', 'A_mode', 'B_mode']
    
        color_features_df = pd.DataFrame(color_features, columns=columns)
    
        filename = output_path / f"{csv_file.stem}_Feature{output_extention}"
        color_features_df.to_csv(filename, index=False)
        print(f"file [{csv_file.name}] color feature has been saved")

def GLCMFeatureWriter(data_path: str, output_path: str, angles: list, distance: list, levels=256, output_extention='.csv'):
    image_files = sorted(list(data_path.glob('*.tif')))
    
    if not image_files:
        print('image directory not found. skipping')
    
    glcm_features = []
    features = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation']
    glcm_angles = ['0', '90']
    
    glcm_feature_label = []
    for feature in features:
        for glcm_angle in glcm_angles:
            glcm_feature_label.append(f'{feature}_{glcm_angle}')
    
    for image_file in image_files:
        image = cv.imread(image_file)
        image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        
        glcm = graycomatrix(image, distance, angles, levels)
        
        image_features = []
        for feature in features:
            props = graycoprops(glcm, feature)
            image_features.append(props)
            
        feature_vector = np.hstack(image_features)
        glcm_df = pd.DataFrame(feature_vector, columns=glcm_feature_label)
        
        filename = output_path / f'{Path(StringSplitter(str(image_file))).name}_GLCM{output_extention}'
        glcm_df.to_csv(filename, index=False)
        print(f"file [{filename.name}] GLCM feature has been saved")
        
def StringSplitter(filename: str):
    main, exception = filename.rsplit('_', 1)
    
    return main
        
def main():
    CSV_DIR = Path('Tea Score Color Data').resolve()
    
    TEA_SCORE = ['Score 1', 'Score 2', 'Score 3', 'Score 4']
    COLOR_MODE = ['RGB', 'HSV', 'LAB']
    
    OUTPUT_DIR = Path('Tea Score Feature Data').resolve()
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    OUTPUT_COLOR_DIR = OUTPUT_DIR / 'Color Feature Data'
    OUTPUT_COLOR_DIR.mkdir(exist_ok=True)
    
    IMAGE_DIR = Path('Preprocessed Tea Score Images').resolve()
    OUTPUT_GLCM_DIR = OUTPUT_DIR / 'GLCM Feature Data'
    OUTPUT_GLCM_DIR.mkdir(exist_ok=True)
    
    start = time()
    
    for score in TEA_SCORE:
        csv_dir = CSV_DIR / score
        image_dir = IMAGE_DIR / score
        
        output_glcm_dir = OUTPUT_GLCM_DIR / score
        output_glcm_dir.mkdir(exist_ok=True)
        
        output_color_dir = OUTPUT_COLOR_DIR / score
        output_color_dir.mkdir(exist_ok=True)
        
        for mode in COLOR_MODE:
            output_color_mode_dir = output_color_dir / f'{score}_{mode}'
            output_color_mode_dir.mkdir(exist_ok=True)
            
            csv_mode_dir = csv_dir / f'{score}_{mode}'
            
            ColorFeatureWriter(csv_mode_dir, mode, output_color_mode_dir)
            
            if mode == 'RGB':
                image_dir = image_dir / f'{score}_{mode}'
                GLCMFeatureWriter(image_dir, output_glcm_dir, [0, np.pi/2], [5])
                
    finish = time()
    time_taken = finish - start
    
    print(f'processing time: {time_taken} seconds')            
    
if __name__ == "__main__":
    main()