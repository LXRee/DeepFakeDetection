import cv2
from PIL import Image
from random import random
import numpy as np


# Source: https://www.kaggle.com/timesler/facial-recognition-model-in-pytorch
class DetectionPipeline:
    """Pipeline class for detecting faces in the frames of a video file."""

    def __init__(self, detector, n_frames=None, resize=None, window=5):
        """Constructor for DetectionPipeline class.

        Keyword Arguments:
            n_frames {int} -- Total number of frames to load. These will be evenly spaced
                throughout the video. If not specified (i.e., None), all frames will be loaded.
                (default: {None})
            batch_size {int} -- Batch size to use with MTCNN face detector. (default: {32})
            resize {float} -- Fraction by which to resize frames from original prior to face
                detection. A value less than 1 results in downsampling and a value greater than
                1 result in upsampling. (default: {None})
            window {int} -- number of frames to skip between each one
        """
        self.detector = detector
        self.n_frames = n_frames
        self.resize = resize
        self.window = window

    def __call__(self, filename):
        """Load frames from an MP4 video and detect faces.

        Arguments:
            filename {str} -- Path to video.
        """
        # Create video reader and find length
        v_cap = cv2.VideoCapture(filename)
        v_len = int(v_cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Pick 'n_frames' consequently frames from video
        if self.n_frames is None:
            start_sample = 0
            stop_sample = v_len
        else:
            start_sample = int(random()*(v_len - self.n_frames * self.window))
            stop_sample = self.n_frames + start_sample

        start_sample = np.clip(start_sample, 0, v_len)
        stop_sample = np.clip(stop_sample, 0, v_len)

        # set initial frame from which to start reading
        v_cap.set(cv2.CAP_PROP_POS_FRAMES, start_sample)
        v_height = int(v_cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * self.resize)
        v_width = int(v_cap.get(cv2.CAP_PROP_FRAME_WIDTH) * self.resize)
        # Loop through frames
        batch_size = (stop_sample - start_sample) // self.window
        frames = np.zeros((batch_size, v_height, v_width, 3), dtype='uint8')

        i = 0
        # frames = []
        while i*self.window < stop_sample - self.window:
            _ = v_cap.grab()  # select next frame
            # Load frame
            success, frame = v_cap.retrieve()
            if not success:
                continue
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (v_width, v_height))
            frames[i] = frame

            i += 1

        faces = self.detector([frames[i] for i in range(len(frames))])

        v_cap.release()

        return faces