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

# Function to preprocess image and get embeddings
def get_image_embedding(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    
    embedding = model.predict(img_array)
    return embedding.flatten()  # Ensure the embedding is flattened to 1D

#feature extracting 

def load_images_from_folder(folder_path):
    X = []
    y = []
    for student_name in os.listdir(folder_path):
        student_folder = os.path.join(folder_path, student_name)
        if os.path.isdir(student_folder):
            for image_name in os.listdir(student_folder):
                image_path = os.path.join(student_folder, image_name)
                try:
                    embedding = get_image_embedding(image_path)
                    X.append(embedding)
                    y.append(student_name)
                except Exception as e:
                    print(f"Could not process image {image_path}: {e}")
    return np.array(X), np.array(y)

def save_features_to_numpy(X, y, feature_file='features.npy', label_file='labels.npy'):
    np.save(feature_file, X)
    np.save(label_file, y)
    print(f"Features saved to {feature_file}")
    print(f"Labels saved to {label_file}")

# Main execution
folder_path = r'E:\crop2\crop2'  # Replace this with your folder path
X, y = load_images_from_folder(folder_path)
save_features_to_numpy(X, y)



