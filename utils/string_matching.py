import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import time

# Load the MobileNetV2 model
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Create a new model that outputs the embeddings
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

# Function to load features and labels from numpy files
def load_features_from_numpy(feature_file='features.npy', label_file='labels.npy'):
    X = np.load(feature_file)
    y = np.load(label_file)
    return X, y

# Function to match face with KNN
def match_face_with_knn(X, y, query_image_path, k=5):
    knn = KNeighborsClassifier(n_neighbors=k, metric='euclidean')
    knn.fit(X, y)
    
    query_embedding = get_image_embedding(query_image_path)
    start_time = time.time()
    predicted_label = knn.predict([query_embedding])
    end_time = time.time()
    
    print("Time taken for prediction:", end_time - start_time)
    return predicted_label

# Load features and labels
X, y = load_features_from_numpy('features.npy', 'labels.npy')

# Path to the query image
query_image_path = r""# Replace this with your query image path

# Match the face and get the predicted label
predicted_label = match_face_with_knn(X, y, query_image_path)
print("Predicted label:", predicted_label[0])
