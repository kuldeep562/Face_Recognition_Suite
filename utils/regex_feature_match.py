from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import numpy as np
import os
#modle loading
# Load the MobileNetV2 model
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Create a new model that outputs the embeddings
# GlobalAveragePooling2D layer will convert the feature maps into a single vector per image
x = base_model.output
x = tf.keras.layers.GlobalAveragePooling2D()(x)
model = Model(inputs=base_model.input, outputs=x) 

def get_image_embedding(img_path, datagen=None):
    # Load and preprocess the image
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    
    # Apply data augmentation if provided
    if datagen:
        img_array = datagen.flow(img_array, batch_size=1).next()
    else:
        img_array = img_array / 255.0  # Normalize if no augmentation
    
    img_array = preprocess_input(img_array)
    
    embedding = model.predict(img_array)
    return embedding.flatten()

datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    brightness_range=[0.5, 1.5],  # Adjust brightness randomly
    preprocessing_function=preprocess_input
)
def load_images_from_folder(folder_path, datagen=None):
    X = []
    y = []
    for student_name in os.listdir(folder_path):
        student_folder = os.path.join(folder_path, student_name)
        if os.path.isdir(student_folder):
            for image_name in os.listdir(student_folder):
                image_path = os.path.join(student_folder, image_name)
                try:
                    embedding = get_image_embedding(image_path, datagen)
                    X.append(embedding)
                    y.append(student_name)
                except Exception as e:
                    print(f"Could not process image {image_path}: {e}")
    return np.array(X), np.array(y)
# Example usage with data augmentation
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    brightness_range=[0.5, 1.5],
    preprocessing_function=preprocess_input
)
def save_features_to_numpy(X, y, feature_file='features.npy', label_file='labels.npy'):
    np.save(feature_file, X)
    np.save(label_file, y)
    print(f"Features saved to {feature_file}")
    print(f"Labels saved to {label_file}")

folder_path = r'E:\crop 1-20'  # Replace with your folder path
X, y = load_images_from_folder(folder_path, datagen)
save_features_to_numpy(X, y)
