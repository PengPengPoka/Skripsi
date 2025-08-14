"""
Feature extraction from RGB, HSV, and LAB images
Splitting the images into individual channel to extract the data
"""

import cv2 as cv
import os

def ImageColorReader(directory, variant, output_dir, extention='.jpg'):
    """
    args:
        directory   : directory containing the files of the tea images
        variant     : variant of tea
        output_dir  : output directory to store the processed images
        extention   : file format for the image, defaulted to jpg
    """
    os.makedirs(output_dir, exist_ok=True)
    print(f"Creating {output_dir} directory")
    


def main():
    directory = os.getcwd()
    DATASET_DIRECTORY = os.path.join(directory, "Processed Images")
    OUTPUT_DIRECTOR = os.path.join(directory, "Colour Data")
    
    TOTAL_IMAGES_PER_VARIANT = [561, 600, 300, 601, 590, 430, 599, 300, 441, 600]
    TEA_VARIANTS = ['BOHEA', 'BOP', 'BOPF', 'DUST', 'DUST_II', 'F_I', 'F_II', 'PF', 'PF_II', 'PF_III']
    
    
    
    
main()