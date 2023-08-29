import threading
import numpy as np
from PIL import Image
from keras import Model

from roop.typing import Frame

PREDICTOR = None
THREAD_LOCK = threading.Lock()
MAX_PROBABILITY = 0.85


def get_predictor() -> None:
    global PREDICTOR

    PREDICTOR = None


def clear_predictor() -> None:
    global PREDICTOR

    PREDICTOR = None


def predict_frame(target_frame: Frame) -> bool:
    image = Image.fromarray(target_frame)
    image = preprocess_image(image)  # Replace this with the code to preprocess your image
    views = np.expand_dims(image, axis=0)
    _, probability = get_predictor().predict(views)[0]
    return probability > MAX_PROBABILITY


def predict_image(target_path: str) -> bool:
    return predict_single_image(target_path) > MAX_PROBABILITY  # Replace this with your image prediction logic


def predict_video(target_path: str) -> bool:
    _, probabilities = predict_video_frames(video_path=target_path, frame_interval=100)  # Replace this with your video prediction logic
    return any(probability > MAX_PROBABILITY for probability in probabilities)
