"""
Cropping image data of tea using mask
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

    # child directory
    # structure of directory
    # parent -> child (Variant) 

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
            cropped_image = cv.bitwise_and(blurred_image, crop_mask)
            cv.imwrite(os.path.join(processed_variant_path, filename), cropped_image)
            print(f"image {filename} has been created at {processed_variant_path}")

        else:
            print(f"Error reading image {filename} at {image_path}")
            failed_images.append(filename)
            continue
    
    print("All images have been processed")
    return failed_images


def main():
    directory = os.getcwd()
    DATASET_DIRECTORY = os.path.join(directory, "Data_NEW")
    OUTPUT_DIRECTORY = os.path.join(directory, "Cropped Images")

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