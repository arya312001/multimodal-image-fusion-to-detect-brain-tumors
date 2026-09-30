import matplotlib.pyplot as plt
from PIL import Image

# Define the image array and labels
image_array = [
    {'image': 'ct.jpg', 'label': 'CT Image'},
    {'image': 'mri_registered.jpg', 'label': 'MRI Image'},
    {'image': 'fusion.jpg', 'label': 'Ground Truth Image'},
    {'image': 'maxmax.jpg', 'label': 'Max-Max'},
    {'image': 'maxmin.jpg', 'label': 'Max-Min'},
    {'image': 'maxmean.jpg', 'label': 'Max-Mean'},
    {'image': 'minmax.jpg', 'label': 'Min-Max'},
    {'image': 'minmin.jpg', 'label': 'Min-Min'},
    {'image': 'minmean.jpg', 'label': 'Min-Mean'},
    {'image': 'meanmax.jpg', 'label': 'Mean-Max'},
    {'image': 'meanmin.jpg', 'label': 'Mean-Min'},
    {'image': 'meanmean.jpg', 'label': 'Mean-Mean'},
    # Add more images and labels as needed
]

# Create the subplot
fig, axes = plt.subplots(4, 3, figsize=(8, 8))

# Iterate over the image array and plot each image
for i, data in enumerate(image_array):
    image_path = data['image']
    label = data['label']
    ax = axes[i // 3, i % 3]  # Get the appropriate subplot axis

    # Load the image using PIL and convert to RGB color space
    image = Image.open(image_path).convert('RGB')

    ax.imshow(image)
    ax.set_title(label)
    ax.axis('off')
    ax.grid(True, color='gray', linewidth=0.5)  # Set the grid color to gray

# Add a heading to the subplot
plt.suptitle('Results of Person 38 with Haar Fusion', fontsize=16, fontweight='bold')

# Adjust the spacing between subplots
plt.subplots_adjust(hspace=0.3, wspace=0.1)
plt.margins(0.05, 0.05)

# Save the plot
plt.savefig('P38_Haar_All_Results.jpg')

# Show the plot
plt.show()
