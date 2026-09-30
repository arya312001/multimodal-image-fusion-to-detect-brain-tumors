from PIL import Image, ImageDraw, ImageFont
import random

# Define the table size
table_width = 10
table_height = 17

# Define cell dimensions
cell_width = 40
cell_height = 20

# Define image dimensions
image_width = table_width * cell_width
image_height = table_height * cell_height

# Create a new image
image = Image.new("RGB", (image_width, image_height), "white")
draw = ImageDraw.Draw(image)

# Define font properties
font = ImageFont.truetype("arial.ttf", 12)

# Draw table borders
for i in range(table_height + 1):
    y = i * cell_height
    draw.line([(0, y), (image_width, y)], fill="black")

for j in range(table_width + 1):
    x = j * cell_width
    draw.line([(x, 0), (x, image_height)], fill="black")

# Generate random values and fill the table
for i in range(table_height):
    for j in range(table_width):
        value = random.randint(1, 100)
        cell_x = j * cell_width
        cell_y = i * cell_height
        cell_text = str(value)
        text_width, text_height = draw.textsize(cell_text, font=font)
        text_x = cell_x + (cell_width - text_width) // 2
        text_y = cell_y + (cell_height - text_height) // 2
        draw.text((text_x, text_y), cell_text, fill="black", font=font)

# Save the image
image.save("table_image.png")
