#Notes: This is meant to run on Google Colab. Before running anything, make sure to upload the .zip file with the whole data set.
#If the name of the zip has changed, just change it as well in the zip_path variable.
import zipfile
import os

zip_path = '/content/archive.zip'
extract_path = '/content/'

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_path)

print(f"Dataset extracted to: {extract_path}")

#After extracting all files from the zip. Running this snippet will help you depict the csv files
import pandas as pd

#load the training and both tests sets
train_df = pd.read_csv('/content/train.csv')
test_df = pd.read_csv('/content/test.csv')
test_v2_df = pd.read_csv('/content/test_v2.csv')

# show the first few rows of the train and test dataset to verify everything is correct
print("Train DataFrame:")
print(train_df.head())

print("Test DataFrame:")
print(test_df.head())

print("Test_v2 DataFrame:")
print(test_v2_df.head())

#This snippets serves for visualizing if the images were uploaded correctly

import os
from PIL import Image
import matplotlib.pyplot as plt

# list the subdirectories in the images folder
image_dir = '/content/images/'
subdirectories = os.listdir(image_dir)

# display a random image from one of the subdirectories
example_subdir = subdirectories[5]  #change the value in [] to see other images
example_image_path = os.path.join(image_dir, example_subdir, os.listdir(os.path.join(image_dir, example_subdir))[0])

# plot the image
image = Image.open(example_image_path)
plt.imshow(image)
plt.title(f"Example Image: {example_image_path}")
plt.show()


#Since our data set consists on pairs of images (from the same place but different time) this cell functionality...
#is to showcase different images in order to compare them visually and verify that everything is in order
import os
from PIL import Image
import matplotlib.pyplot as plt
import random

def load_image_pair(row, image_dir='/content/images/'):
    # Construct file paths based on 'dir' and 'img_1', 'img_2'
    img_1_path = os.path.join(image_dir, row['dir'], row['img_1'])
    img_2_path = os.path.join(image_dir, row['dir'], row['img_2'])

    # Load the images
    img_1 = Image.open(img_1_path)
    img_2 = Image.open(img_2_path)

    return img_1, img_2

# Select a random row index
random_index = random.randint(0, len(train_df) - 1)

# Get the row corresponding to the selected index
random_row = train_df.iloc[random_index]

# Load the images
img_1, img_2 = load_image_pair(random_row)

# Plot the images side by side
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].imshow(img_1)
ax[0].set_title(f"Image 1: {random_row['img_1']}")
ax[1].imshow(img_2)
ax[1].set_title(f"Image 2: {random_row['img_2']}")
plt.show()


#this snippet plots the images from the previous cell but after filtering using NDVI values
import numpy as np

def calculate_ndvi(image):
    # Convert image to numpy array
    image_np = np.array(image) / 255.0  # Normalize to 0-1 range
    # Calculate NDVI using Green (index 1) and Red (index 0) channels
    ndvi = (image_np[:, :, 1] - image_np[:, :, 0]) / (image_np[:, :, 1] + image_np[:, :, 0] + 1e-10)
    return ndvi

# Calculate NDVI for the first image
ndvi_img_1 = calculate_ndvi(img_1)
ndvi_img_2 = calculate_ndvi(img_2)

# Visualize the NDVI images
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].imshow(ndvi_img_1, cmap='RdYlGn')
ax[0].set_title("NDVI Image 1")
ax[1].imshow(ndvi_img_2, cmap='RdYlGn')
ax[1].set_title("NDVI Image 2")
plt.show()

#white color in image represents the deforestation or change in vegetal density (from more to less)
#calculate the absolute difference between NDVI images
ndvi_diff = np.abs(ndvi_img_2 - ndvi_img_1)

#set a threshold for deforestation detection (e.g., difference above 0.1 indicates change)
threshold = 0.1
deforestation_mask = (ndvi_diff > threshold).astype(np.uint8)

#visualize the change (deforestation) mask
plt.imshow(deforestation_mask, cmap='gray')
plt.title("Deforestation Mask (Change Detection)")
plt.show()
