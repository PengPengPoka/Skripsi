import cv2 as cv
import numpy as np
import pandas as pd
from pathlib import Path
from time import time

def FeatureExtraction(data_dir: str, output_path: str, color_mode=['RGB', 'HSV', 'LAB'], output_extention='.csv'):
    csv_files = sorted(list(data_dir.glob('*.tif')))

    failed_images = []
    roi_data_df = pd.DataFrame()

    if not csv_files:
        print(f"directory '{data_dir}' not found. skipping")
        return failed_images
    
    print(f"processing '{len(csv_files)}' image files in [{data_dir}]")

    for csv_file in csv_files:
        img = cv.imread(csv_file)
        if img is not None:
            print(f"processing '{csv_file.name}' image")
            
            if color_mode == 'RGB':
                img_roi = getROIData(img)
                roi_data_df = pd.DataFrame(img_roi, columns=['B','G','R'])
            elif color_mode == 'HSV':
                img_roi = getROIData(img)
                roi_data_df = pd.DataFrame(img_roi, columns=['H','S','V'])
            elif color_mode == 'LAB':
                img_roi = getROIData(img)
                roi_data_df = pd.DataFrame(img_roi, columns=['L','A','B'])

        else:
            print(f"image file [{csv_file.name}] could not be read. skipping")
            failed_images.append(data_dir)
            continue

        if roi_data_df is not None:
            # csv_filename = output_path / f"{tea_variant}_{color_mode}_{''.join(img_id)}.csv"
            csv_filename = Path(csv_file.name).stem
            output_file = output_path / f"{csv_filename}{output_extention}"
            roi_data_df.to_csv(output_file, index=False)
            print(f"file [{output_file}] saved")

    return failed_images

def getROIData(src):
    gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
    ret, mask = cv.threshold(gray, 1, 255, cv.THRESH_BINARY)
    roi_coor = np.where(mask == 255)
    roi_data = src[roi_coor]

    return roi_data

def main():
    # for all data
    # DATA_DIR = Path('Tea Score Images - partition').resolve() / 'Preprocessed partition images' 
    # OUTPUT_DIR = Path('Tea Score Color Data')
    # OUTPUT_DIR.mkdir(exist_ok=True)

    # for partition data
    ROOT_DIR = Path('Tea Score Images - partition').resolve()
    DATA_DIR = ROOT_DIR / 'Preprocessed partition images'
    OUTPUT_DIR = ROOT_DIR / 'Partition color data'
    OUTPUT_DIR.mkdir(exist_ok=True)

    TEA_SCORE = ['Score 1', 'Score 2', 'Score 3', 'Score 4']
    COLOR_MODE = ['RGB', 'HSV', 'LAB']
    
    start_time = time()

    for score in TEA_SCORE:
        score_dir = DATA_DIR / score

        output_score_dir = OUTPUT_DIR / score
        output_score_dir.mkdir(exist_ok=True)

        for mode in COLOR_MODE:
            score_mode_dir = score_dir / f"{score}_{mode}"

            output_score_mode_dir = output_score_dir / f"{score}_{mode}"
            output_score_mode_dir.mkdir(exist_ok=True)
    
            failed_images = FeatureExtraction(score_mode_dir,
                                              output_score_mode_dir,
                                              mode)
    
    end_time = time()
    process_time = end_time - start_time

    print(f"process done in {process_time:.3f} seconds")
    print(f"number of failed images: {len(failed_images)}")
    print(failed_images)

if __name__ == "__main__":
    main()