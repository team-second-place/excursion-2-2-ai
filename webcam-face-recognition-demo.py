from asyncio import create_task, run, sleep, to_thread
from collections import defaultdict, deque
from dataclasses import dataclass
from enum import auto, Enum

# import io
# import os
from threading import Thread
from typing import Any, Generic, Optional, TypeVar

import cv2
import face_recognition

# import matplotlib
# import matplotlib.pyplot as plt
import numpy as np

# import pandas as pd
# import PIL
# from PIL import Image, ImageDraw, ImageFont
# import scipy.misc

from emitter import Event, Listenable, emittable, listenable, mapped
from emitter.utils import latest
from store import derived, get, writable

requested_quit = emittable()

is_training_store = writable(None)


def emit_from_webcam(cam: cv2.VideoCapture, emit):
    while True:
        # print("about to read from camera")
        ret, frame = cam.read()
        # print("done reading from camera")

        # await sleep(1/30)
        # await sleep(0)

        key = cv2.waitKey(1) % 256

        ESCAPE = 27
        if key == ESCAPE:
            print("quitting")
            requested_quit.emit(())
            return

        ONE = ord("1")
        TWO = ord("2")
        THREE = ord("3")
        FOUR = ord("4")

        if key == ONE:
            is_training_store.update(lambda now: 1 if now != 1 else None)
        elif key == TWO:
            is_training_store.update(lambda now: 2 if now != 2 else None)
        elif key == THREE:
            is_training_store.update(lambda now: 3 if now != 3 else None)
        elif key == FOUR:
            is_training_store.update(lambda now: 4 if now != 4 else None)

        # print(f"{key=} {chr(key)=}")

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


# workaround to deal with np arrays being uncomparable
Wrapped = TypeVar("Wrapped")


@dataclass(eq=False)
class Wrapper(Generic[Wrapped]):
    value: Wrapped

    def __eq__(self, other):
        return self is other


Classification = type(None)  # TODO
Frame = np.ndarray

Data = tuple[Wrapper[Frame], Classification]


font = cv2.FONT_HERSHEY_COMPLEX


def draw(data: Data):
    # print(f"{data=}")
    (frame, faces) = data
    # print(f"{frame=}")
    # frame: Frame = data

    # if frame is None:
    #     cv2.namedWindow('Webcam Feed')

    if frame is not None:
        frame = frame.value
        image = frame.copy()

        if faces is not None:
            for face in faces:
                text = ""
                color = GRAY  # only used for malfunction

                match face.state:
                    case FaceState.NO_FACES_YET:
                        color = WHITE
                    case FaceState.TOO_MANY_WHILE_TRAINING:
                        text = "fewer people please"
                        color = GRAY

                    case FaceState.SUCCESFULLY_TRAINING | FaceState.COULD_MATCH:
                        match face.matches:
                            case 1:
                                color = RED
                            case 2:
                                color = YELLOW
                            case 3:
                                color = GREEN
                            case 4:
                                color = BLUE
                            # oh no! intruder!
                            case None:
                                color = BLACK

                match face.state:
                    case FaceState.SUCCESFULLY_TRAINING:
                        text = f"training {face.matches}"
                    case FaceState.COULD_MATCH:
                        if face.matches is None:
                            text = "INTRUDER!!!!!"
                        else:
                            text = f"matched {face.matches}"

                cv2.rectangle(
                    image,
                    (face.location[3], face.location[0]),
                    (face.location[1], face.location[2]),
                    color,
                    2,
                )

                cv2.putText(
                    image,
                    text,
                    (face.location[3], face.location[0]),
                    font,
                    1.5,
                    color,
                    2,
                    cv2.LINE_AA,
                )

        cv2.imshow("Webcam Feed", image)


WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (0, 0, 255)
YELLOW = (0, 255, 255)
GREEN = (0, 255, 0)
BLUE = (255, 128, 0)
GRAY = (128, 128, 128)


def collect(*args) -> tuple:
    return args


trusted_faces = defaultdict(list)


class FaceState(Enum):
    NO_FACES_YET = auto()
    SUCCESFULLY_TRAINING = auto()
    TOO_MANY_WHILE_TRAINING = auto()
    COULD_MATCH = auto()


@dataclass
class DetectedFace:
    location: Any
    matches: Optional[int]
    state: FaceState


faces_store = writable([])


def find_faces_as_often_as_possible(latest_image_store):
    while True:
        latest_image = get(latest_image_store)

        if latest_image is None:
            continue

        latest_image = latest_image.value

        face_locations = face_recognition.face_locations(latest_image)

        is_training = get(is_training_store)

        if is_training is None and not trusted_faces:
            faces_store.set(
                [
                    DetectedFace(
                        location=face_location,
                        matches=None,
                        state=FaceState.NO_FACES_YET,
                    )
                    for face_location in face_locations
                ]
            )
            continue

        face_encodings = face_recognition.face_encodings(latest_image)

        if is_training is not None:
            if len(face_locations) == 1:
                assert len(face_encodings) == 1
                trusted_faces[is_training].append(face_encodings[0])
                faces_store.set(
                    [
                        DetectedFace(
                            location=face_location,
                            matches=is_training,
                            state=FaceState.SUCCESFULLY_TRAINING,
                        )
                        for face_location in face_locations
                    ]
                )
            else:
                print("there is more than one face in frame while training - ignoring")
                faces_store.set(
                    [
                        DetectedFace(
                            location=face_location,
                            matches=is_training,
                            state=FaceState.TOO_MANY_WHILE_TRAINING,
                        )
                        for face_location in face_locations
                    ]
                )
        else:
            faces = []

            for face_location, face_encoding in zip(face_locations, face_encodings):
                for face_id, trusted_face_encodings in trusted_faces.items():
                    matches = face_recognition.compare_faces(
                        trusted_face_encodings, face_encoding
                    )

                    match = any(matches)

                    if match:
                        faces.append(
                            DetectedFace(
                                location=face_location,
                                matches=face_id,
                                state=FaceState.COULD_MATCH,
                            )
                        )

                        break
                else:
                    faces.append(
                        DetectedFace(
                            location=face_location,
                            matches=None,
                            state=FaceState.COULD_MATCH,
                        )
                    )

            faces_store.set(faces)


async def main():
    camera_stream = listenable(stream_from_webcam)

    latest_image_store = latest(mapped(camera_stream, Wrapper))

    data_store = derived([latest_image_store, faces_store], collect)

    stop_showing = data_store.subscribe(draw)

    requested_quit.listen(lambda _event: stop_showing())
    requested_quit.listen(lambda _event: exit(0))

    await sleep(0)

    find_faces_thread = Thread(
        target=find_faces_as_often_as_possible, args=(latest_image_store,)
    )
    find_faces_thread.start()


if __name__ == "__main__":
    run(main())
