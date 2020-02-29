from random import random

import cv2
import numpy as np
import torch

from data_preparation.helper.custom_exceptions import NoFrames
from data_preparation.helper.make_chunks import list_chunks


# Heavily modified source: https://www.kaggle.com/timesler/facial-recognition-model-in-pytorch

class DetectionPipeline:
    """Pipeline class for detecting faces in the frames of a video file."""

    def __init__(self, detector, n_frames=None, resize=None, window=5):
        """
        Constructor for DetectionPipeline class.

        @:param n_frames {int}: Total number of frames to load. These will be evenly spaced
                throughout the video. If not specified (i.e., None), all frames will be loaded.
                (default: {None})
        @:param batch_size {int}: Batch size to use with MTCNN face detector. (default: {32})
        @:param resize {float}: Fraction by which to resize frames from original prior to face
                detection. A value less than 1 results in downsampling and a value greater than
                1 result in upsampling. (default: {None})
        @:param window {int}: number of frames to skip between each one
        """
        self.detector = detector
        self.n_frames = n_frames
        self.resize = resize
        self.window = window

    def __call__(self, filename):
        """
        Load frames from an MP4 video and detect faces.

        @:param filename {str}: Path to video.
        """
        # Create video reader and find length
        v_cap = cv2.VideoCapture(filename)
        v_len = int(v_cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # In case of missing video file just skip and and raise exception
        if v_len == 0:
            raise NoFrames

        # Pick 'n_frames' subsequent frames from video
        if self.n_frames is None:
            start_sample = 0
            stop_sample = v_len
        else:
            # Take into consideration window length
            start_sample = int(random() * (v_len - self.n_frames * self.window))
            stop_sample = self.n_frames * self.window + start_sample

        # Avoid indexes checking by clamping start and stop length
        start_sample = np.clip(start_sample, 0, v_len)
        stop_sample = np.clip(stop_sample, 0, v_len)

        # Set initial frame from which to start reading
        v_cap.set(cv2.CAP_PROP_POS_FRAMES, start_sample)

        # Define frame height and width to prepare an empty numpy vector
        v_height = int(v_cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * self.resize)
        v_width = int(v_cap.get(cv2.CAP_PROP_FRAME_WIDTH) * self.resize)

        # Batch size is the quantity of images that are going to be kept
        batch_size = (stop_sample - start_sample) // self.window

        # Prepare emtpy batch of frames
        frames = np.zeros((batch_size, v_height, v_width, 3), dtype='uint8')

        # Loop through frames
        i = 0
        while i * self.window < stop_sample - self.window:
            # Select next frame
            _ = v_cap.grab()

            # Load frame
            success, frame = v_cap.retrieve()

            if not success:
                continue

            # Decode and resize frame
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (v_width, v_height))

            # Put frame into batch of frames
            frames[i] = frame

            # Index update
            i += 1

        # Clean CUDA cache every loop in order to process also high-bitrate videos in parallel with other processes.
        # Keep a list of images as long as possible, since it is pretty faster.
        # If there is memory problem, just decrease "permitted_length".
        permitted_length = 10
        chunks = list_chunks(frames, permitted_length)
        faces = []
        for frame in chunks:
            # Detect faces in list of frames
            faces.extend(self.detector([frame[i] for i in range(len(frame))]))
            # Free useless cache. Every step is independent from one another.
            torch.cuda.empty_cache()

        # Release video hook
        v_cap.release()

        return faces
