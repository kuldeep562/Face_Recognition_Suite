import cv2
import numpy as np
import os
from openvino.runtime import Core
from multiprocessing import Pool
import time

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

def draw_bboxes(image, bboxes, scores, color=[0, 255, 0]):
    for box, score in zip(bboxes, scores):
        if score > 0.95:
            x_min, y_min, x_max, y_max = box
            pt1 = (int(x_min), int(y_min))  # Convert coordinates to integers
            pt2 = (int(x_max), int(y_max))  # Convert coordinates to integers
            cv2.rectangle(image, pt1, pt2, color=color, thickness=2, lineType=cv2.LINE_4)
            cv2.putText(image, f"{score:.2f}", (int(x_min), int(y_min) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

def face_detection(frame, model_path):
    core = Core()
    model = core.read_model(model=model_path)
    compiled_model = core.compile_model(model=model)

    executable_network = compiled_model.load_network(core, 'CPU')

    input_image = preprocess_image(frame)
    pred_scores = executable_network.infer(inputs={executable_network.input_info['data'].input_name: input_image})['detection_out']
    pred_boxes = executable_network.infer(inputs={executable_network.input_info['data'].input_name: input_image})['detection_out']

    image_shape = frame.shape[:2]
    faces, scores = postprocess(pred_scores, pred_boxes, image_shape)
    return faces, scores

def process_frame(args):
    frame, model_path = args
    faces, scores = face_detection(frame, model_path)
    return faces, scores

if __name__ == '__main__':
    vid_path = r'vid_path'

    if not os.path.exists(vid_path):  
        print('File {} not found'.format(vid_path))
        exit(-1)

    a = cv2.VideoCapture(vid_path)

    model_path = r'model_path'
    num_processes = 4
    pool = Pool(num_processes)

    while True:
        start_time = time.time()

        frames = []
        for _ in range(num_processes):
            success, frame = a.read()
            if success:
                frames.append(frame)

        if not frames:
            break

        args_list = [(frame, model_path) for frame in frames]
        results = pool.map(process_frame, args_list)

        all_faces = []
        all_scores = []
        for faces, scores in results:
            all_faces.extend(faces)
            all_scores.extend(scores)

        for frame, faces, scores in zip(frames, all_faces, all_scores):
            draw_bboxes(frame, faces, scores)
            cv2.imshow('show', frame)
            k = cv2.waitKey(1)
            if k == ord('q'):
                break

        print("FPS:", 1.0 / (time.time() - start_time))

    pool.close()
    pool.join()

    cv2.destroyAllWindows()
