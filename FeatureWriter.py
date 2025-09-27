import pandas as pd
import numpy as np
from pathlib import Path

def FeatureWriter(data_path):
    ...

def main():
    MASTER_DIR = Path('Color Data').resolve()

    TEA_VARIANTS = ['BOHEA', 'BOP', 'BOPF', 'DUST', 'DUST_II', 'F_I', 'F_II', 'PF', 'PF_II', 'PF_III']
    COLOR_MODE = ['RGB', 'HSV', 'LAB']
    
    OUTPUT_DIR = MASTER_DIR / 'Feature Data'
    OUTPUT_DIR.mkdir(exist_ok=True
                     )
    
if __name__ == "__main__":
    main()