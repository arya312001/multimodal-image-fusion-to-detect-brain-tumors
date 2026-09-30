import cv2
import numpy as np

# Function to compare evaluation parameters
def compare_images(image1, image2):
    # Compute MSE
    mse = np.mean((image1 - image2) ** 2)

    # Compute SSIM
    ssim = cv2.compareSSIM(image1, image2)

    # Compute PSNR
    psnr = cv2.PSNR(image1, image2)

    return mse, ssim, psnr

# Specify the paths of the two images
image_path1 = "path/to/first/image.jpg"
image_path2 = "path/to/second/image.jpg"

# Load the input images
image1 = cv2.imread(image_path1)
image2 = cv2.imread(image_path2)

# Resize the images
resized_image1 = cv2.resize(image1, (256, 256))
resized_image2 = cv2.resize(image2, (256, 256))

# Compare the resized images
mse, ssim, psnr = compare_images(resized_image1, resized_image2)

# Display the results
print("Mean Squared Error (MSE):", mse)
print("Structural Similarity Index (SSIM):", ssim)
print("Peak Signal-to-Noise Ratio (PSNR):", psnr)
