import cv2 as cv
import numpy as np
import pandas as pd
from pathlib import Path
from time import time

from Preprocessing_new import getImageID

def FeatureExtraction(data_dir: str, output_path: str, tea_variant: str, color_mode=['RGB', 'HSV', 'LAB']):
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

            img_id = getImageID(csv_file)
            
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
            csv_filename = output_path / f"{tea_variant}_{color_mode}_{''.join(img_id)}.csv"
            roi_data_df.to_csv(csv_filename, index=False)
            print(f"file [{csv_filename}] saved")

    return failed_images

def getROIData(src):
    gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
    ret, mask = cv.threshold(gray, 0, 255, cv.THRESH_BINARY)
    roi_coor = np.where(mask == 255)
    roi_data = src[roi_coor]

    return roi_data

def main():
    DATA_DIR = Path('Preprocessed Images').absolute().resolve()
    
    OUTPUT_DIR = Path('Color Data').absolute().resolve()
    OUTPUT_DIR.mkdir(exist_ok=True)

    TEA_VARIANTS = ['BOHEA', 'BOP', 'BOPF', 'DUST', 'DUST_II', 'F_I', 'F_II', 'PF', 'PF_II', 'PF_III']
    COLOR_MODE = ['RGB', 'HSV', 'LAB']
    
    start_time = time()

    for variant in TEA_VARIANTS:
        variant_dir = DATA_DIR / variant

        output_variant_dir = OUTPUT_DIR / variant
        output_variant_dir.mkdir(exist_ok=True)

        for mode in COLOR_MODE:
            variant_mode_dir = variant_dir / f"{variant}_{mode}"

            output_variant_mode_dir = output_variant_dir / f"{variant}_{mode}"
            output_variant_mode_dir.mkdir(exist_ok=True)
    
            failed_images = FeatureExtraction(variant_mode_dir, output_variant_mode_dir,
                                              variant, mode)
    
    end_time = time()
    process_time = end_time - start_time

    print(f"process done in {process_time:.3f} seconds")
    print(f"number of failed images: {len(failed_images)}")
    print(failed_images)

if __name__ == "__main__":
    main()