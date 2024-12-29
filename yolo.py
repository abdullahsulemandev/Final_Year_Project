import cv2
import numpy as np
from ultralytics import YOLO as YOLOv8  # Import YOLOv8
from database import log_detection
from collections import defaultdict
import os

# Function to resize image to 416x416 while maintaining the aspect ratio by padding
def resize_with_padding(image, target_size=(416, 416), pad_color=(0, 0, 0)):
    original_height, original_width = image.shape[:2]
    target_width, target_height = target_size

    # Calculate scaling factors to fit the image in the target size
    scaling_factor = min(target_width / original_width, target_height / original_height)
    new_width = int(original_width * scaling_factor)
    new_height = int(original_height * scaling_factor)

    # Resize the image while maintaining the aspect ratio
    resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

    # Calculate padding needed to fill the remaining space
    top_pad = (target_height - new_height) // 2
    bottom_pad = target_height - new_height - top_pad
    left_pad = (target_width - new_width) // 2
    right_pad = target_width - new_width - left_pad

    # Pad the resized image with the pad_color
    padded_image = cv2.copyMakeBorder(resized_image, top_pad, bottom_pad, left_pad, right_pad, 
                                      cv2.BORDER_CONSTANT, value=pad_color)

    return padded_image

# Function to process and resize images in a folder
def process_images(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    # Loop through all files in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(input_folder, filename)

            # Read the image using OpenCV
            image = cv2.imread(image_path)

            # Resize the image to 416x416 while maintaining aspect ratio by padding
            padded_image = resize_with_padding(image, (416, 416))

            # Create base filename without extension and add .png
            base_filename = os.path.splitext(filename)[0]

            # Save the resized image as PNG
            resized_filename = os.path.join(output_folder, f"{base_filename}.png")
            cv2.imwrite(resized_filename, padded_image)

# Processing example images (path updated)
input_folder = 'TOAugment/more validation/other'
output_folder = 'TOAugment/more validation/shit'

process_images(input_folder, output_folder)


# Centroid Tracker class for object tracking
class CentroidTracker:
    def __init__(self, max_disappeared=50):
        self.next_object_id = 0
        self.objects = {}
        self.disappeared = defaultdict(int)
        self.max_disappeared = max_disappeared

    def register(self, centroid):
        self.objects[self.next_object_id] = centroid
        self.next_object_id += 1

    def deregister(self, object_id):
        del self.objects[object_id]
        del self.disappeared[object_id]

    def update(self, rects):
        if len(rects) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return self.objects

        input_centroids = np.zeros((len(rects), 2), dtype="int")

        for (i, (startX, startY, endX, endY)) in enumerate(rects):
            cX = int((startX + endX) / 2.0)
            cY = int((startY + endY) / 2.0)
            input_centroids[i] = (cX, cY)

        if len(self.objects) == 0:
            for i in range(0, len(input_centroids)):
                self.register(input_centroids[i])
        else:
            object_ids = list(self.objects.keys())
            object_centroids = list(self.objects.values())

            D = np.linalg.norm(np.array(object_centroids)[:, np.newaxis] - input_centroids, axis=2)
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            used_rows = set()
            used_cols = set()

            for (row, col) in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue

                object_id = object_ids[row]
                self.objects[object_id] = input_centroids[col]
                self.disappeared[object_id] = 0

                used_rows.add(row)
                used_cols.add(col)

            unused_rows = set(range(0, D.shape[0])).difference(used_rows)
            unused_cols = set(range(0, D.shape[1])).difference(used_cols)

            if D.shape[0] >= D.shape[1]:
                for row in unused_rows:
                    object_id = object_ids[row]
                    self.disappeared[object_id] += 1
                    if self.disappeared[object_id] > self.max_disappeared:
                        self.deregister(object_id)
            else:
                for col in unused_cols:
                    self.register(input_centroids[col])

        return self.objects


# YOLO Class with updated command
class YOLO:
    def __init__(self):
        # Path to your custom model
        custom_model_path = 'best.pt'

        # Initialize YOLOv8 with your custom model
        self.model = YOLOv8('yolov8s.pt')
        self.classes = self.model.names  # Get class names from YOLOv8 model
        self.tracker = CentroidTracker()
        self.detected_boxes = []
        self.logged_objects = set()  # Set to keep track of logged object IDs

    def detect(self, frame):
        results = self.model(frame)  # YOLOv8 inference
        detections = results[0].boxes.xyxy.cpu().numpy()  # Get detections
        confidences = results[0].boxes.conf.cpu().numpy()  # Get confidence scores
        class_ids = results[0].boxes.cls.cpu().numpy()  # Get class IDs

        self.detected_boxes = []

        rects = []
        for i, (startX, startY, endX, endY) in enumerate(detections):
            conf = confidences[i]
            class_index = int(class_ids[i])
            if class_index < 0 or class_index >= len(self.classes):
                continue
            rects.append([startX, startY, endX, endY])

        objects = self.tracker.update(rects)

        for i, (startX, startY, endX, endY) in enumerate(detections):
            conf = confidences[i]
            class_index = int(class_ids[i])
            if class_index < 0 or class_index >= len(self.classes):
                continue

            centroid = ((startX + endX) / 2, (startY + endY) / 2)
            object_id = None
            for id, tracked_centroid in objects.items():
                if np.allclose(centroid, tracked_centroid, atol=1.0):
                    object_id = id
                    break

            class_name = self.classes[class_index]
            label = f'{class_name} {conf:.2f} ID: {object_id}'
            frame = cv2.rectangle(frame, (int(startX), int(startY)), (int(endX), int(endY)), (255, 0, 0), 2)
            frame = cv2.putText(frame, label, (int(startX), int(startY) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            if object_id not in self.logged_objects:
                log_detection(class_name, conf, startX, startY, endX, endY, object_id)
                self.logged_objects.add(object_id)

        return frame

# Example of running YOLOv8 from command
# !yolo predict model='C:/Users/Sharjeel Pathan/Downloads/train7/train7/weights/best.pt' source='TOAugment/more validation/other' conf=0.25 save=True