import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logging
import time
import tensorflow as tf
from absl import app, flags, logging
from absl.flags import FLAGS
from core import utils
from core.yolov4 import filter_boxes
from core.functions import *
from tensorflow.python.saved_model import tag_constants
from PIL import Image
import cv2
import numpy as np
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
import streamlit as st
import tempfile
import sys

# Configuration flags
flags.DEFINE_string('framework', 'tf', '(tf, tflite, trt)')
flags.DEFINE_string('weights', './checkpoints/yolov4-tiny-416', 'path to weights file')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_boolean('tiny', True, 'yolo or yolo-tiny')
flags.DEFINE_string('model', 'yolov4', 'yolov3 or yolov4')
flags.DEFINE_string('video', './data/video/video.mp4', 'path to input video')
flags.DEFINE_string('output', 'output.mp4', 'path to output video')
flags.DEFINE_string('output_format', 'mp4v', 'codec used in VideoWriter')
flags.DEFINE_float('iou', 0.45, 'iou threshold')
flags.DEFINE_float('score', 0.50, 'score threshold')
flags.DEFINE_boolean('count', False, 'count objects within video')
flags.DEFINE_boolean('dont_show', False, 'dont show video output')
flags.DEFINE_boolean('info', False, 'print info on detections')
flags.DEFINE_boolean('crop', False, 'crop detections from images')
flags.DEFINE_boolean('plate', False, 'perform license plate recognition')

DEMO_VIDEO = 'new.mp4'


def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image

    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    return cv2.resize(image, dim, interpolation=inter)


def main(_argv):
    # Streamlit UI Setup
    st.title('Custom Object Detection')
    st.sidebar.title('Configuration')

    # GPU configuration
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)

    # User inputs
    use_webcam = st.sidebar.checkbox('Use Webcam')
    confidence = st.sidebar.slider('Confidence', 0.0, 1.0, 0.5)
    video_file_buffer = st.sidebar.file_uploader("Upload a video", type=["mp4", "mov", "avi"])

    # Video source selection
    tffile = tempfile.NamedTemporaryFile(delete=False)
    if not video_file_buffer:
        vid = cv2.VideoCapture(0 if use_webcam else DEMO_VIDEO)
        tffile.name = DEMO_VIDEO
    else:
        tffile.write(video_file_buffer.read())
        vid = cv2.VideoCapture(tffile.name)

    # Model loading
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)

    STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(FLAGS)
    input_size = FLAGS.size

    if FLAGS.framework == 'tflite':
        interpreter = tf.lite.Interpreter(model_path=FLAGS.weights)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
    else:
        saved_model_loaded = tf.saved_model.load(FLAGS.weights, tags=[tag_constants.SERVING])
        infer = saved_model_loaded.signatures['serving_default']

    # Video writer setup
    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(vid.get(cv2.CAP_PROP_FPS))

    # Use MP4V codec instead of VP8 for better compatibility
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(FLAGS.output, fourcc, fps, (width, height))

    # Streamlit display
    stframe = st.empty()
    stop_button = st.sidebar.button('Stop Processing')

    # Main processing loop
    frame_num = 0
    while vid.isOpened() and not stop_button:
        return_value, frame = vid.read()
        if not return_value:
            st.warning("Video ended or failed to read frame")
            break

        # Preprocessing
        frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame)
        frame_size = frame.shape[:2]

        # Model inference
        image_data = cv2.resize(frame, (input_size, input_size))
        image_data = image_data / 255.
        image_data = image_data[np.newaxis, ...].astype(np.float32)
        start_time = time.time()

        if FLAGS.framework == 'tflite':
            interpreter.set_tensor(input_details[0]['index'], image_data)
            interpreter.invoke()
            pred = [interpreter.get_tensor(output_details[i]['index']) for i in range(len(output_details))]
        else:
            batch_data = tf.constant(image_data)
            pred_bbox = infer(batch_data)
            for key, value in pred_bbox.items():
                boxes = value[:, :, 0:4]
                pred_conf = value[:, :, 4:]

        # Post-processing
        boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
            boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
            scores=tf.reshape(pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
            max_output_size_per_class=50,
            max_total_size=50,
            iou_threshold=FLAGS.iou,
            score_threshold=confidence
        )

        # Draw bounding boxes
        original_h, original_w, _ = frame.shape
        bboxes = utils.format_boxes(boxes.numpy()[0], original_h, original_w)
        pred_bbox = [bboxes, scores.numpy()[0], classes.numpy()[0], valid_detections.numpy()[0]]

        
        class_names = utils.read_class_names(cfg.YOLO.CLASSES)
        image = utils.draw_bbox(frame, pred_bbox, FLAGS.info, allowed_classes=list(class_names.values()))

        # Display and save
        fps = 1.0 / (time.time() - start_time)
        st.sidebar.text(f"FPS: {fps:.2f}")

        result = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        out.write(result)
        result = image_resize(result, width=720)
        stframe.image(result, channels='BGR', use_container_width=True)

    # Cleanup
    vid.release()
    out.release()

    # Show output video
    st.success('Processing complete!')
    with open(FLAGS.output, 'rb') as video_file:
        st.video(video_file.read())


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass