from asyncio import create_task, run, sleep, to_thread
from collections import deque
from dataclasses import dataclass
import io
import os
from threading import Thread
from typing import Generic, TypeVar

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import PIL
from PIL import Image, ImageDraw, ImageFont
import scipy.misc
from six import BytesIO
import tensorflow as tf
import tensorflow_datasets as tfds

from object_detection.utils import label_map_util
from object_detection.utils import config_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder

from emitter import Event, Listenable, emittable, listenable, mapped
from emitter.utils import latest
from store import derived, derived_unchecked, writable, writable_unchecked


def load_image_into_numpy_array(path):
    """Load an image from file into a numpy array.

    Puts image into numpy array to feed into tensorflow graph.
    Note that by convention we put it into a numpy array with shape
    (height, width, channels), where channels=3 for RGB.

    Args:
        path: the file path to the image

    Returns:
        uint8 numpy array with shape (img_height, img_width, 3)
    """
    img_data = tf.io.gfile.GFile(path, 'rb').read()
    image = Image.open(BytesIO(img_data))
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)

def get_keypoint_tuples(eval_config):
    """Return a tuple list of keypoint edges from the eval config.
    
    Args:
        eval_config: an eval config containing the keypoint edges
    
    Returns:
        a list of edge tuples, each in the format (start, end)
    """
    tuple_list = []
    kp_list = eval_config.keypoint_edge
    for edge in kp_list:
        tuple_list.append((edge.start, edge.end))
    return tuple_list


# @title Choose the model to use, then evaluate the cell.
MODELS = {'centernet_with_keypoints': 'centernet_hg104_512x512_kpts_coco17_tpu-32', 'centernet_without_keypoints': 'centernet_hg104_512x512_coco17_tpu-8'}

model_display_name = 'centernet_with_keypoints' # @param ['centernet_with_keypoints', 'centernet_without_keypoints']
model_name = MODELS[model_display_name]


# pipeline_config = os.path.join('models/research/object_detection/configs/tf2/',
#                                 model_name + '.config')
pipeline_config = 'models/research/object_detection/configs/tf2/centernet_hourglass104_512x512_coco17_tpu-8.config'
model_dir = 'models/research/object_detection/test_data/checkpoint/'

# Load pipeline config and build a detection model
configs = config_util.get_configs_from_pipeline_file(pipeline_config)
model_config = configs['model']
detection_model = model_builder.build(
      model_config=model_config, is_training=False)

# Restore checkpoint
ckpt = tf.compat.v2.train.Checkpoint(
      model=detection_model)
ckpt.restore(os.path.join(model_dir, 'ckpt-0')).expect_partial()

def get_model_detection_function(model):
    """Get a tf.function for detection."""
    
    @tf.function
    def detect_fn(image):
        """Detect objects in image."""

        image, shapes = model.preprocess(image)
        prediction_dict = model.predict(image, shapes)
        detections = model.postprocess(prediction_dict, shapes)

        return detections, prediction_dict, tf.reshape(shapes, [-1])

    return detect_fn

detect_fn = get_model_detection_function(detection_model)


# label_map_path = configs['eval_input_config'].label_map_path
label_map_path = 'models/research/object_detection/data/mscoco_label_map.pbtxt'
label_map = label_map_util.load_labelmap(label_map_path)
categories = label_map_util.convert_label_map_to_categories(
    label_map,
    max_num_classes=label_map_util.get_max_label_map_index(label_map),
    use_display_name=True)
category_index = label_map_util.create_category_index(categories)
label_map_dict = label_map_util.get_label_map_dict(label_map, use_display_name=True)

requested_quit = emittable()

# Things to try:
# Flip horizontally
# image_np = np.fliplr(image_np).copy()

# Convert image to grayscale
# image_np = np.tile(
#     np.mean(image_np, 2, keepdims=True), (1, 1, 3)).astype(np.uint8)


def emit_from_webcam(cam: cv2.VideoCapture, emit):
    # print(f"{cam=} {emit=}")

    while True:
        # print("about to read from camera")
        ret, frame = cam.read()
        # print("done reading from camera")

        # await sleep(1/30)
        # await sleep(0)

        key = cv2.waitKey(1) % 256
        # print("done waiting for key press")
        ESCAPE = 27
        if key == ESCAPE:
            print("quitting")
            requested_quit.emit(())
            return

        if ret:
            # print("emitting frame", frame)
            # print("emitting frame")
            emit(frame)
        else:
            print("not ret", frame)


def stream_from_webcam(emit):
    cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    emit_task = create_task(to_thread(emit_from_webcam, cam, emit))

    def stop():
        print("stopping")
        emit_task.cancel()
        cam.release()
        cam.destroyAllWindows()

    return stop

def every_n(emitter: Listenable[Event], n: int) -> Listenable[Event]:
    def start(emit):
        i = n - 1 # include the first frame (for demo purposes)

        def handler(event):
            nonlocal i
            i += 1

            if i == n:
                i = 0
                emit(event)

        unlisten = emitter.listen(handler)
        return unlisten

    return listenable(start)

# workaround to deal with np arrays being uncomparable
Wrapped = TypeVar("Wrapped")
@dataclass(eq=False)
class Wrapper(Generic[Wrapped]):
    value: Wrapped
    
    def __eq__(self, other):
        return self is other



Classification_store = type(None) # TODO
Frame = np.ndarray

Data = tuple[Wrapper[Frame], Classification_store]


font = cv2.FONT_HERSHEY_COMPLEX

def draw(data: Data):
    # print(f"{data=}")
    (frame, classification) = data
    # print(f"{frame=}")
    # frame: Frame = data

    # if frame is None:
    #     cv2.namedWindow('Webcam Feed')

    if frame is not None:
        # from PIL import Image
        frame = frame.value
        image = frame.copy()
        # cv2.line(image, (0, 0), (511, 511), (255, 0, 0), 5)
        # cv2.rectangle(image, (384, 0), (510, 128), (0, 255, 0), 3)

        image_shape = image.shape
        # print(image_shape)

        if classification is not None:
            # cv2.putText(
            #     image, "OpenCV", (10, 500), font, 4, (255, 255, 255), 2, cv2.LINE_AA
            # )

            (
                detections,
                keypoints,
                keypoint_scores,
                label_id_offset,
            ) = classification.value
            
            viz_utils.visualize_boxes_and_labels_on_image_array(
                image,
                detections['detection_boxes'][0].numpy(),
                (detections['detection_classes'][0].numpy() + label_id_offset).astype(int),
                detections['detection_scores'][0].numpy(),
                category_index,
                use_normalized_coordinates=True,
                max_boxes_to_draw=200,
                min_score_thresh=.30,
                agnostic_mode=False,
                keypoints=keypoints,
                keypoint_scores=keypoint_scores,
                keypoint_edges=get_keypoint_tuples(configs['eval_config']))

        cv2.imshow("Webcam Feed", image)

def collect(*args) -> tuple:
    return args

async def main():
    labels_path = tf.keras.utils.get_file(
        "imagenet-labels.txt",
        "https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt"
    )

    labels = np.array(open(labels_path).readlines())

    camera_stream = listenable(stream_from_webcam)

    # TODO: run the model here
    # camera_stream.listen(calculate_stuff)

    # camera_stream.listen(print)

    # latest_image = writable(np.ndarray(shape=()))
    latest_image = writable(None)
    latest_image = latest(mapped(camera_stream, Wrapper))

    classification_store = writable(None)

    def classify(frame: Frame):
        image_np = frame

        input_tensor = tf.convert_to_tensor(
            np.expand_dims(image_np, 0), dtype=tf.float32)
        print("running detect function")
        detections, _predictions_dict, _shapes = detect_fn(input_tensor)
        print("after running detect function")

        label_id_offset = 1

        # Use keypoints if available in detections
        keypoints, keypoint_scores = None, None
        if 'detection_keypoints' in detections:
            keypoints = detections['detection_keypoints'][0].numpy()
            keypoint_scores = detections['detection_keypoint_scores'][0].numpy()

        classification_store.set(
            Wrapper((
                detections,
                keypoints,
                keypoint_scores,
                label_id_offset,
            ))
        )
    


    every_many_frames = every_n(camera_stream, 480)


    # threads: deque[Thread] = deque()
    def add_classification_thread(frame):
        thread = Thread(target=classify, args=(frame,))

        # if len(threads) > 4:
        #     oldest_thread = threads.popleft()

        #     oldest_thread.

        # threads.push(thread)
        thread.start()

    every_many_frames.listen(add_classification_thread)

    data_store = derived([latest_image, classification_store], collect)

    stop_showing = data_store.subscribe(draw)

    requested_quit.listen(lambda _event: stop_showing())
    requested_quit.listen(lambda _event: exit(0))


if __name__ == "__main__":
    run(main())
