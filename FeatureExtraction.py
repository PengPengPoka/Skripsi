import cv2 as cv
import numpy as np
import pandas as pd
from pathlib import Path

def FeatureExtraction():
    ...

def main():
    BASE_DIR = Path(__file__).absolute().parent
    PREPROCESSED_DATA_DIR = BASE_DIR / "Preprocessed Images"
    
    COLOR_MODE = ['RGB', 'HSV', 'LAB']
    TEA_VARIANTS = ['BOHEA', 'BOP', 'BOPF', 'DUST', 'DUST_II', 'F_I', 'F_II', 'PF', 'PF_II', 'PF_III']
    
    
    
if __name__ == "__main__":
    main()