import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from sklearn.utils import class_weight


def computeNDVI(image1, image2):
    # Considering the third channel is the Red band (index 2 in RGB)
    RED1 = image1[:, :, 2]
    RED2 = image2[:, :, 2]

    # For this case, we will assume the green channel is being approximated as NIR
    NIR1 = image1[:, :, 1]  # ideally you'd have the NIR band
    NIR2 = image2[:, :, 1]  # Same for image 2

    # Calculate NDVI for both images
    ndvi1 = (NIR1 - RED1) / (NIR1 + RED1 + 1e-7)  # a small value is added to avoid division by zero
    ndvi2 = (NIR2 - RED2) / (NIR2 + RED2 + 1e-7)

    return ndvi1, ndvi2


def loadAndPreprocessImagesWithNDVI(df, imageDir='/content/images', imgSize=(224, 224)):
    images = []
    labels = []

    for _, row in df.iterrows():
        img1Path = os.path.join(imageDir, row['dir'], row['img_1'])
        img2Path = os.path.join(imageDir, row['dir'], row['img_2'])

        # Load the two images
        img1 = load_img(img1Path, target_size=imgSize)
        img2 = load_img(img2Path, target_size=imgSize)

        # Convert them to arrays and normalize
        img1Array = img_to_array(img1) / 255.0  # Shape: (224, 224, 3)
        img2Array = img_to_array(img2) / 255.0  # Shape: (224, 224, 3)

        # Compute NDVI for both images
        ndvi1, ndvi2 = computeNDVI(img1Array, img2Array)

        # Stack NDVI as additional channels
        img1NDVI = np.expand_dims(ndvi1, axis=-1)  # Shape: (224, 224, 1)
        img2NDVI = np.expand_dims(ndvi2, axis=-1)  # Shape: (224, 224, 1)

        # Concatenate along the channel axis (last axis) to include NDVI
        imgPair = np.concatenate([img1Array, img2Array, img1NDVI, img2NDVI], axis=-1)  # Shape: (224, 224, 8)

        images.append(imgPair)
        labels.append(row['label'])

    return np.array(images), np.array(labels)



# Load the datasets with NDVI filtering
X_train, y_train = loadAndPreprocessImagesWithNDVI(train_df)
X_test, y_test = loadAndPreprocessImagesWithNDVI(test_v2_df)


# Function to extract meaningful part of the label
def extract_label(label):
    return label.split('\\')[0]  # Adjust this based on the actual delimiter in your file path if necessary

# Apply label extraction to both train and test sets
train_df['label'] = train_df['label'].apply(extract_label)
test_v2_df['label'] = test_v2_df['label'].apply(extract_label)

# Create a label map and encode the labels
labelMap = {label: idx for idx, label in enumerate(np.unique(train_df['label']))}

# Relabel the training and test labels
y_train = np.array([labelMap[label] for label in train_df['label']])
y_test = np.array([labelMap[label] if label in labelMap else -1 for label in test_v2_df['label']])  # Assign -1 for unknown labels

# Split training data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

from sklearn.utils.class_weight import compute_class_weight
import numpy as np

# Calculate class weights using the labels from the training set
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)

# Convert to a dictionary where the key is the class index, and the value is the weight
class_weights_dict = {i: weight for i, weight in enumerate(class_weights)}

print("Computed class weights:", class_weights_dict)

from tensorflow.keras import models, layers

# Model creation function
def create_model_with_ndvi(input_shape=(224, 224, 8)):  # 8 channels now: RGB x 2 + NDVI x 2
    model = models.Sequential([
        layers.InputLayer(input_shape=input_shape),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(len(np.unique(y_train)), activation='softmax')  # Output layer based on class count
    ])

    # Compile the model
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    return model


# Create the model
model = create_model_with_ndvi()

# Train the model with class weights
history = model.fit(X_train, y_train,
                    validation_data=(X_val, y_val),
                    class_weight=class_weights_dict,  # Pass the computed class weights
                    epochs=10)


test_loss, test_acc = model.evaluate(X_test, y_test)
print(f'Test accuracy: {test_acc:.4f}, Test loss: {test_loss:.4f}')


model.save('reforestation_model.h5')


def visualize_predictions(model, X_test, y_test, labelMap):
    # Randomly select a few test images
    num_samples = 5
    sample_indices = np.random.choice(len(X_test), num_samples, replace=False)

    plt.figure(figsize=(15, 3*num_samples))

    for i, idx in enumerate(sample_indices):
        # Extract original images and NDVI channels
        img1 = X_test[idx][:,:,:3]  # First RGB image
        img2 = X_test[idx][:,:,3:6]  # Second RGB image
        ndvi1 = X_test[idx][:,:,6]   # First NDVI channel
        ndvi2 = X_test[idx][:,:,7]   # Second NDVI channel

        # Make prediction
        prediction = model.predict(X_test[idx:idx+1])
        predicted_class = np.argmax(prediction)
        true_class = y_test[idx]

        # Plotting
        plt.subplot(num_samples, 4, i*4+1)
        plt.imshow(img1)
        plt.title(f"Image 1")
        plt.axis('off')

        plt.subplot(num_samples, 4, i*4+2)
        plt.imshow(img2)
        plt.title(f"Image 2")
        plt.axis('off')

        plt.subplot(num_samples, 4, i*4+3)
        plt.imshow(ndvi1, cmap='viridis')
        plt.title(f"NDVI 1")
        plt.colorbar()

        plt.subplot(num_samples, 4, i*4+4)
        plt.imshow(ndvi2, cmap='viridis')
        plt.title(f"NDVI 2")
        plt.colorbar()

        # Add prediction information
        print(f"Sample {i+1}:")
        print(f"Predicted Class: {list(labelMap.keys())[list(labelMap.values()).index(predicted_class)]}")
        print(f"True Class: {list(labelMap.keys())[list(labelMap.values()).index(true_class)]}")

    plt.tight_layout()
    plt.show()

    visualize_predictions(model, X_test, y_test, labelMap)



    ##########
    import os
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt

# Function to compute NDVI
def computeNDVI(image1, image2):
    RED1 = image1[:, :, 2]  # Red channel for image 1
    RED2 = image2[:, :, 2]  # Red channel for image 2

    NIR1 = image1[:, :, 1]  # Assuming NIR is approximated by the green channel for now
    NIR2 = image2[:, :, 1]

    # NDVI calculation
    ndvi1 = (NIR1 - RED1) / (NIR1 + RED1 + 1e-7)
    ndvi2 = (NIR2 - RED2) / (NIR2 + RED2 + 1e-7)

    return ndvi1, ndvi2

# Function to load and preprocess two satellite images
def load_and_preprocess_user_images(image1_path, image2_path, img_size=(224, 224)):
    # Load the two images
    img1 = load_img(image1_path, target_size=img_size)
    img2 = load_img(image2_path, target_size=img_size)

    # Convert them to arrays and normalize
    img1_array = img_to_array(img1) / 255.0  # Shape: (224, 224, 3)
    img2_array = img_to_array(img2) / 255.0  # Shape: (224, 224, 3)

    # Compute NDVI for both images
    ndvi1, ndvi2 = computeNDVI(img1_array, img2_array)

    # Stack NDVI as additional channels
    img1_ndvi = np.expand_dims(ndvi1, axis=-1)  # Shape: (224, 224, 1)
    img2_ndvi = np.expand_dims(ndvi2, axis=-1)  # Shape: (224, 224, 1)

    # Concatenate along the channel axis (last axis) to include NDVI
    img_pair = np.concatenate([img1_array, img2_array, img1_ndvi, img2_ndvi], axis=-1)  # Shape: (224, 224, 8)

    return np.expand_dims(img_pair, axis=0), ndvi1, ndvi2  # Add batch dimension for prediction and return NDVIs

# Function to plot the original images and their NDVI versions
def plot_images_with_ndvi(image1, image2, ndvi1, ndvi2):
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))

    # Plot original images
    axes[0, 0].imshow(image1)
    axes[0, 0].set_title("Original Image 1 (Earlier Date)")
    axes[0, 0].axis('off')

    axes[0, 1].imshow(image2)
    axes[0, 1].set_title("Original Image 2 (Later Date)")
    axes[0, 1].axis('off')

    # Plot NDVI images (with colormap for better visualization)
    ndvi1_plot = axes[1, 0].imshow(ndvi1, cmap='RdYlGn')
    axes[1, 0].set_title("NDVI Image 1")
    axes[1, 0].axis('off')
    fig.colorbar(ndvi1_plot, ax=axes[1, 0])

    ndvi2_plot = axes[1, 1].imshow(ndvi2, cmap='RdYlGn')
    axes[1, 1].set_title("NDVI Image 2")
    axes[1, 1].axis('off')
    fig.colorbar(ndvi2_plot, ax=axes[1, 1])

    plt.show()

# Function to make predictions on the provided images
def make_prediction(model, img_pair):
    prediction = model.predict(img_pair)
    predicted_class = np.argmax(prediction, axis=-1)
    return predicted_class

# Main function for user input, prediction, and plotting
def main():
    # Get user input for the two image paths
    image1_path = input("Enter the path to the first satellite image (earlier date): ")
    image2_path = input("Enter the path to the second satellite image (later date): ")

    # Check if the files exist
    if not os.path.exists(image1_path) or not os.path.exists(image2_path):
        print("One or both image paths are invalid. Please check and try again.")
        return

    # Preprocess the images and get their NDVIs
    img_pair, ndvi1, ndvi2 = load_and_preprocess_user_images(image1_path, image2_path)

    # Load the AI model
    model = create_model_with_ndvi()

    # Make the prediction
    predicted_class = make_prediction(model, img_pair)

    # Print the predicted class (you can adjust this to map the class to a meaningful label)
    print(f"The model predicts the class as: {predicted_class[0]}")

    # Load original images for plotting
    img1 = load_img(image1_path)
    img2 = load_img(image2_path)

    # Plot the original images with their NDVI filters
    plot_images_with_ndvi(img_to_array(img1) / 255.0, img_to_array(img2) / 255.0, ndvi1, ndvi2)

if __name__ == "__main__":
    main()
