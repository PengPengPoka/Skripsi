"""
Application of 5x5 median blur to original image
Conversion of original image into RGB, HSV, and LAB color space
Cropping the image with a circular mask
"""

import cv2 as cv
import os
import time

def ColorConverter(directory, variant, total_images, output_dir, crop_mask, extention='.jpg'):
    """
    args:
        directory   : directory containing the files of the tea images
        variant     : variant of tea
        total_images: total images in a variant
        output_dir  : output directory to store the processed images
        crop_masl   : the mask used for cropping the image
        extention   : file format for the image, defaulted to jpg
    """

    os.makedirs(output_dir, exist_ok=True)
    print(f"Creating {output_dir} directory")

    # parent directory | location of processed data
    processed_variant_path = os.path.join(output_dir, variant)

    # child directory | location of variants and their respective colour space
    # structure of directory
    # parent -> child (Variant) -> child_RGB
    #                           -> child_HSV
    #                           -> child_LAB
    RGB_path = os.path.join(processed_variant_path, f"{variant}_RGB")
    HSV_path = os.path.join(processed_variant_path, f"{variant}_HSV")
    LAB_path = os.path.join(processed_variant_path, f"{variant}_LAB")
    
    # creation of child directories
    os.makedirs(RGB_path, exist_ok=True)
    os.makedirs(HSV_path, exist_ok=True)
    os.makedirs(LAB_path, exist_ok=True)    

    failed_images = []

    print(f"--- Processing of {variant} in directory {directory} ---")
    
    for i in range(1, total_images + 1):
        image_id = f"{i:03d}"
        filename = f"{variant}_{image_id}{extention}"
        image_path = os.path.join(directory, filename)

        image = cv.imread(image_path)

        if image is not None:
            print(f"{filename} image loaded")

            blurred_image = cv.medianBlur(image, 5)             # blurring by median blur (5x5)

            rgb = cv.cvtColor(blurred_image, cv.COLOR_BGR2RGB)  # conversion to RGB
            hsv = cv.cvtColor(blurred_image, cv.COLOR_BGR2HSV)  # conversion to HSV
            lab = cv.cvtColor(blurred_image, cv.COLOR_BGR2Lab)  # conversion to lab

            # cropping using mask
            cropped_rgb = cv.bitwise_and(rgb, crop_mask)
            cropped_hsv = cv.bitwise_and(hsv, crop_mask)
            cropped_lab = cv.bitwise_and(lab, crop_mask)

            # saving images into respective directories
            cv.imwrite(os.path.join(RGB_path, f"{variant}_RGB_{image_id}{extention}"), cropped_rgb)
            cv.imwrite(os.path.join(HSV_path, f"{variant}_HSV_{image_id}{extention}"), cropped_hsv)
            cv.imwrite(os.path.join(LAB_path, f"{variant}_LAB_{image_id}{extention}"), cropped_lab)

        else:
            print(f"Error reading image {filename} at {image_path}")
            failed_images.append(filename)
            continue
    
    print("All images have been processed")
    return failed_images


def main():
    directory = os.getcwd()
    DATASET_DIRECTORY = os.path.join(directory, "Data_NEW")
    OUTPUT_DIRECTORY = os.path.join(directory, "Preprocessed Images")

    TOTAL_IMAGES_PER_VARIANT = [561, 600, 300, 601, 590, 430, 599, 300, 441, 600]
    TEA_VARIANTS = ['BOHEA', 'BOP', 'BOPF', 'DUST', 'DUST_II', 'F_I', 'F_II', 'PF', 'PF_II', 'PF_III']

    mask = cv.imread(directory + "\\Testing Results\\mask.jpg")

    start = time.time()

    for variant, images_per_variant in zip(TEA_VARIANTS, TOTAL_IMAGES_PER_VARIANT):
        variant_directory = os.path.join(DATASET_DIRECTORY, variant)

        failed_list = ColorConverter(variant_directory,
                                     variant,
                                     images_per_variant,
                                     OUTPUT_DIRECTORY,
                                     mask)

    end = time.time()
    processing_time = end-start

    print(f"Time taken for processing {processing_time:2.0f} seconds")
    print(f"Images failed to be processed: {failed_list}")


main()