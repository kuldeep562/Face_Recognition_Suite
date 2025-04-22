import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import cv2
import os
import time
from openvino.runtime import Core

# Load the MobileNetV2 model for face recognition
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
x = base_model.output
x = tf.keras.layers.GlobalAveragePooling2D()(x)
embedding_model = Model(inputs=base_model.input, outputs=x)

def get_image_embedding(img):
    img_array = cv2.resize(img, (224, 224))
    img_array = image.img_to_array(img_array)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    
    embedding = embedding_model.predict(img_array)
    return embedding.flatten()

def load_features_from_numpy(feature_file='features.npy', label_file='labels.npy'):
    X = np.load(feature_file)
    y = np.load(label_file)
    return X, y

# Load the KNN model with stored features and labels
X, y = load_features_from_numpy('features.npy', 'labels.npy')
knn = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
knn.fit(X, y)

def preprocess_image(image):
    input_image = cv2.resize(image, dsize=[320, 240])
    input_image = np.expand_dims(input_image.transpose(2, 0, 1), axis=0)
    return input_image

def postprocess(pred_scores, pred_boxes, image_shape, confidence_thr=0.5, overlap_thr=0.7):
    filtered_indexes = np.argwhere(pred_scores[0, :, 1] > confidence_thr).reshape(-1)
    filtered_boxes = pred_boxes[0, filtered_indexes, :]
    filtered_scores = pred_scores[0, filtered_indexes, 1]

    if len(filtered_scores) == 0:
        return [], []

    h, w = image_shape
    def convert_bbox_format(bbox):
        x_min, y_min, x_max, y_max = bbox
        x_min = int(w * x_min)
        y_min = int(h * y_min)
        x_max = int(w * x_max)
        y_max = int(h * y_max)
        return x_min, y_min, x_max, y_max

    bboxes_image_coord = np.apply_along_axis(convert_bbox_format, axis=1, arr=filtered_boxes)

    bboxes_with_scores = np.concatenate((bboxes_image_coord, np.expand_dims(filtered_scores, axis=1)), axis=1)

    def calculate_overlap(box, other_boxes):
        x_min = np.maximum(box[0], other_boxes[:, 0])
        y_min = np.maximum(box[1], other_boxes[:, 1])
        x_max = np.minimum(box[2], other_boxes[:, 2])
        y_max = np.minimum(box[3], other_boxes[:, 3])

        intersection_area = np.maximum(0, x_max - x_min + 1) * np.maximum(0, y_max - y_min + 1)
        box_area = (box[2] - box[0] + 1) * (box[3] - box[1] + 1)
        other_boxes_area = (other_boxes[:, 2] - other_boxes[:, 0] + 1) * (other_boxes[:, 3] - other_boxes[:, 1] + 1)

        overlap_ratios = intersection_area / (box_area + other_boxes_area - intersection_area)

        return overlap_ratios

    def non_max_suppression(boxes):
        if len(boxes) == 0:
            return []

        sorted_indices = np.argsort(boxes[:, 4])[::-1]

        picked_indices = []
        while len(sorted_indices) > 0:
            current_index = sorted_indices[0]
            picked_indices.append(current_index)

            current_box = boxes[current_index]
            other_boxes = boxes[sorted_indices[1:]]
            overlap_ratios = calculate_overlap(current_box, other_boxes)

            remaining_indices = np.where(overlap_ratios <= overlap_thr)[0]
            sorted_indices = sorted_indices[remaining_indices + 1]

        return [boxes[i] for i in picked_indices]

    bboxes_after_nms = non_max_suppression(bboxes_with_scores)
    faces = np.array(bboxes_after_nms)[:, :4]
    scores = np.array(bboxes_after_nms)[:, 4]

    return faces, scores

def draw_bboxes(image, bboxes, scores, labels, color=[0, 255, 0]):
    output_path=r"output_path"
    for box, score, label in zip(bboxes, scores, labels):
        if score > 0.5:
            x_min, y_min, x_max, y_max = box
            pt1 = (int(x_min), int(y_min))
            pt2 = (int(x_max), int(y_max))
            cv2.rectangle(image, pt1, pt2, color=color, thickness=2, lineType=cv2.LINE_4)
            cv2.putText(image, f"{str(1)}: {score:.2f}", (int(x_min), int(y_min) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            cv2.imwrite(output_path+f"\\{str(x_min)}_1.jpg", image)

model_path = r'model_path'
core = Core()
model = core.read_model(model=model_path)
compiled_model = core.compile_model(model=model)

def face_detection(frame):
    output_scores_layer = compiled_model.output(0)
    output_boxes_layer = compiled_model.output(1)
    input_image = preprocess_image(frame)
    start_time = time.time()
    pred_scores = compiled_model([input_image])[output_scores_layer]
    pred_boxes = compiled_model([input_image])[output_boxes_layer]
    print("Detection time:", time.time() - start_time)

    image_shape = frame.shape[:2]
    faces, scores = postprocess(pred_scores, pred_boxes, image_shape)
    return faces, scores

def recognize_faces(frame, faces):
    labels = []
    for (x_min, y_min, x_max, y_max) in faces:
        # Ensure coordinates are integers
        x_min, y_min, x_max, y_max = map(int, (x_min, y_min, x_max, y_max))
        face_img = frame[y_min:y_max, x_min:x_max]
        face_embedding = get_image_embedding(face_img)
        label = knn.predict([face_embedding])[0]
        labels.append(label)
    return labels


# Main code
vid_path = r'video_path'
if not os.path.exists(vid_path):
    print('File {} not found'.format(vid_path))
    exit(-1)

prev_frame_time = 0
a = cv2.VideoCapture(vid_path)

while True:
    success, frame = a.read()
    if success:
        faces, scores = face_detection(frame)
        labels = recognize_faces(frame, faces)
        draw_bboxes(frame, faces, scores, labels)

        new_frame_time = time.time()
        if prev_frame_time > 0:
            fps = 1 / (new_frame_time - prev_frame_time)
        else:
            fps=0
        prev_frame_time = new_frame_time

        fps_int = int(fps)
        fps_str = str(fps_int)
        cv2.putText(frame, fps_str, (7, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow('show', frame)
        if cv2.waitKey(1) == ord('q'):
            break

cv2.destroyAllWindows()
