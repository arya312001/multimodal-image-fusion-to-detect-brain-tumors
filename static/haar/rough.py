import cv2
import numpy as np
import os

# Function to compare evaluation parameters
def compare_images(image1, image2):
    # Compute MSE
    mse = np.mean((image1 - image2) ** 2)

    # Compute SSIM
    ssim = cv2.compareSSIM(image1, image2)

    # Compute PSNR
    psnr = cv2.PSNR(image1, image2)

    return mse, ssim, psnr

# Specify the desired dimensions for resizing
new_width = 640
new_height = 480

# Loop through the folders
for folder_num in range(1, 39):
    folder_path = fr"C:\Users\astha\Desktop\multimodal-image-fusion-to-detect-brain-tumors-20230322T120220Z-001\multimodal-image-fusion-to-detect-brain-tumors\static\haar\p{folder_num}"

    # Check if the folder exists
    if os.path.isdir(folder_path):
        # Fetch all image files in the folder
        image_files = os.listdir(folder_path)

        # Loop through the image files
        for image_file in image_files:
            image_path = os.path.join(folder_path, image_file)

            # Load the image
            image = cv2.imread(image_path)

            # Resize the image
            resized_image = cv2.resize(image, (256, 256))

            # Compare the resized image with a reference image
            reference_image = cv2.imread("path/to/reference_image.jpg")
            mse, ssim, psnr = compare_images(resized_image, reference_image)

            # Display the results
            print("Evaluation parameters for", image_path)
            print("MSE:", mse)
            print("SSIM:", ssim)
            print("PSNR:", psnr)
            print("------------------")
