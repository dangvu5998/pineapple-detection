# Import packages
import os
import cv2
import numpy as np
import tensorflow as tf
import sys
import time
from tqdm import tqdm

# Import utilites
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

# Name of the directory containing the object detection module we're using
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_NAME = '/media/trivu/data/DataScience/ComputerVision/dua/data/test_demo/test1.jpeg'

PATH_TO_CKPT = os.path.join(BASE_DIR, 'data/checkpoints/frozen_inference_graph.pb')

# Path to label map file
PATH_TO_LABELS = os.path.join(BASE_DIR, 'data/pineapple_label_map.pbtxt')

# Number of classes the object detector can identify
NUM_CLASSES = 1

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# Load the Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    sess = tf.Session(graph=detection_graph)

# Define input and output tensors (i.e. data) for the object detection classifier

# Input tensor is the image
image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

# Output tensors are the detection boxes, scores, and classes
# Each box represents a part of the image where a particular object was detected
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

# Each score represents level of confidence for each of the objects.
# The score is shown on the result image, together with the class label.
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

# Number of objects detected
num_detections = detection_graph.get_tensor_by_name('num_detections:0')

# Load image using OpenCV and
# expand image dimensions to have shape: [1, None, None, 3]
# i.e. a single-column array, where each item in the column has the pixel RGB value
# for img_fn in os.listdir(PATH_TO_IMAGE_DIRECTORY)[:4]:
image = cv2.imread(IMAGE_NAME)
image_expanded = np.expand_dims(image, axis=0)

# Perform the actual detection by running the model with the image as input
t1 = time.time()
(boxes, scores, classes, num) = sess.run(
    [detection_boxes, detection_scores, detection_classes, num_detections],
    feed_dict={image_tensor: image_expanded})

# Draw the results of the detection (aka 'visulaize the results')
t2 = time.time()
print(t2-t1)
# print('boxes', boxes)
# print('classes', classes)
# print('scores', scores)
# print(np.sum(scores > 0.8))
# print('num', num)
vis_util.visualize_boxes_and_labels_on_image_array(
    image,
    np.squeeze(boxes),
    np.squeeze(classes).astype(np.int32),
    np.squeeze(scores),
    category_index,
    use_normalized_coordinates=True,
    line_thickness=8,
    min_score_thresh=0.80)
# cv2.imwrite('/media/trivu/data/DataScience/CV/dua/data/visualize_test/'+img_fn, image)

# All the results have been drawn on image. Now display the image.
if image.shape[0] > 900:
    image = cv2.resize(image, None, fx=0.5, fy=0.5)
cv2.imshow('Object detector', image)

# Press any key to close the image
cv2.waitKey(0)

# Clean up
cv2.destroyAllWindows()
