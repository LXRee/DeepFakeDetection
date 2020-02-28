import torch
import cv2
from random import random
import numpy as np
from source.data_preparation.helper.custom_exceptions import NoFrames
from source.data_preparation.helper.make_chunks import list_chunks


# Heavily modified source: https://www.kaggle.com/timesler/facial-recognition-model-in-pytorch
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
        if v_len == 0:
            # in case of missing video file just skip and and raise exception
            raise NoFrames

        # Pick 'n_frames' consequently frames from video
        if self.n_frames is None:
            start_sample = 0
            stop_sample = v_len
        else:
            start_sample = int(random()*(v_len - self.n_frames * self.window))  # take into consideration window length
            stop_sample = self.n_frames * self.window + start_sample

        # avoid indexes checking by clamping start and stop length
        start_sample = np.clip(start_sample, 0, v_len)
        stop_sample = np.clip(stop_sample, 0, v_len)

        # set initial frame from which to start reading
        v_cap.set(cv2.CAP_PROP_POS_FRAMES, start_sample)

        # define frame height and width to prepare an empty numpy vector
        v_height = int(v_cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * self.resize)
        v_width = int(v_cap.get(cv2.CAP_PROP_FRAME_WIDTH) * self.resize)

        # batch size is the quantity of images that are going to be kept
        batch_size = (stop_sample - start_sample) // self.window

        # prepare emtpy batch of frames
        frames = np.zeros((batch_size, v_height, v_width, 3), dtype='uint8')

        # Loop through frames
        i = 0
        while i*self.window < stop_sample - self.window:
            # select next frame
            _ = v_cap.grab()

            # Load frame
            success, frame = v_cap.retrieve()

            if not success:
                continue

            # decode and resize frame
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (v_width, v_height))

            # put frame into batch of frames
            frames[i] = frame

            # index update
            i += 1

        # Clean cuda cache every loop in order to process also high-bitrate videos in parallel with other processes.
        # Keep a list of images as long as possible, since it is quite faster.
        # If there is memory problem, just decrease "permitted_length".
        permitted_length = 10
        chunks = list_chunks(frames, permitted_length)
        faces = []
        for frame in chunks:
            # detect faces in list of frames
            faces.extend(self.detector([frame[i] for i in range(len(frame))]))
            # Free useless cache. Every step is independent from one another.
            torch.cuda.empty_cache()

        # release video hook
        v_cap.release()

        return faces
