from asyncio import create_task, run, sleep, to_thread
from dataclasses import dataclass
from threading import Thread
from typing import Generic, TypeVar

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import PIL
from PIL import Image
import tensorflow as tf
import tensorflow_datasets as tfds

from emitter import Event, Listenable, emittable, listenable, mapped
from emitter.utils import latest
from store import derived, derived_unchecked, writable, writable_unchecked


requested_quit = emittable()


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
        i = 0

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
    (frame, classification_store) = data
    # print(f"{frame=}")
    # frame: Frame = data

    # if frame is None:
    #     cv2.namedWindow('Webcam Feed')

    if frame is not None:
        # from PIL import Image
        frame = frame.value
        image = frame.copy()
        # cv2.line(image, (0, 0), (511, 511), (255, 0, 0), 5)
        cv2.rectangle(image, (384, 0), (510, 128), (0, 255, 0), 3)

        image_shape = image.shape
        # print(image_shape)

        if classification_store is not None:
            cv2.putText(
                image, "OpenCV", (10, 500), font, 4, (255, 255, 255), 2, cv2.LINE_AA
            )

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


    mobilenet = tf.keras.applications.MobileNetV2()
    model = mobilenet

    classification_store = writable(None)

    def classify(frame: Frame):

        model_size = (224, 224)

        image = Image.fromarray(frame)

        resized_image = image.resize(model_size)

        resized_image.save("webcam-still.jpg")

        print(np.array(resized_image).shape)

        # correct_shape = (1, model_size[0], model_size[1], 3)

        # correctly_shaped_image_array = np.array(resized_image).reshape(correct_shape)

        correctly_shaped_image_array = np.array(resized_image)[tf.newaxis, ...]
        print(correctly_shaped_image_array.shape)

        preprocessed_array = tf.keras.applications.mobilenet_v2.preprocess_input(correctly_shaped_image_array)

        # frame_array = frame.reshape((1))
        print(f"about to wait for predictions")
        # predictions = await to_thread(model, frame)
        # predictions = await sleep(0)
        # predictions = await sleep(0.5)

        predictions = model(preprocessed_array)

        print(f"{predictions.shape=}")

        argsorted_predictions = np.argsort(predictions)
        # top_5_classes_indices = argsorted_predictions[0,::-1][:5] + 1
        top_5_classes_indices = argsorted_predictions[0,::-1][:5] + 1

        print(top_5_classes_indices)

        top_5_classes = labels[top_5_classes_indices]
        print(top_5_classes)

        classification_store.set(Wrapper(top_5_classes))

    every_20_frames = every_n(camera_stream, 20)
    # every_20_frames.listen(print)
    # TODO: uncomment
    every_20_frames.listen(lambda frame: Thread(target=classify, args=(frame,)).start())

    data_store = derived([latest_image, classification_store], collect)

    # data_store.subscribe(lambda _: print("something"))
    # data_store = latest_image

    stop_showing = data_store.subscribe(draw)
    # stop_showing = camera_stream.listen(draw)

    requested_quit.listen(lambda _event: stop_showing())


if __name__ == "__main__":
    run(main())
