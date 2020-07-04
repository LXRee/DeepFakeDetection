This repository contains my implementation of the [DeepFake Kaggle competition](https://www.kaggle.com/c/deepfake-detection-challenge/).

# Intuition
The approach I tried in this competition comes from a human-perspective attitude: for us (humans) it is way simpler to recognize a fake video from many seconds of playing, instead of from some video frames taken with random probability. Therefore, I tried to extract relevant information from the video frames, and analyze them with a Recurrent Neural Network.

I took inspiration from some notebooks about [data preparation](https://www.kaggle.com/phunghieu/deepfake-detection-data-preparation-baseline), the [training](https://www.kaggle.com/phunghieu/deepfake-detection-training-baseline) loop and the [extraction of audio features](https://www.kaggle.com/cookiecs/resnext-audio-video/)

# Training steps
## Dataset insigth
The dataset contains many `mp4` videos. Those can have the face swapped, audio manipulations, or both. The label associated with the file only states if the video is real o fake (binary label), with no other information.

## Video manipulation
From each video a frame every six is selected and, for each frame, I collect the face (using a `MTCNN` pre-trained network). From each face I extract the features with an `InceptionResNet`. 

Therefore, I save this information on disk with a `pandas` `Dataframe`, which is composed of the columns `filename`, `video_embedding` and `label`. See files `FaceDetectionPipeline.py` and `create_video_embeddings.py` for reference.

## Audio manipulation
I extract the audio track from each video (see file `extract_audio.py`), and I extract the audio histogram, which is then analyzed with a pre-trained CNN.

I save this information inside another `pandas` file, with `filename` and `audio_embedding` columns.

The video and audio embeddings are then merged in one file (`merge_embeddings.py`).

## Network
The network I realized takes at leas a `RANDOM_CROP` number of frames that are then offered to a `LSTM` module. The features that comes with its last timestep are then concatenated with the audio ones. Then, the output follows. In this sense, this fully connected layers takes care of understanding which one among the audio and video are fakes.

# Insights about the project
This project took me a very long time in order to fully understand the task and to select my preferred way to tame this challenge. I immediately encountered an overfitting problem, with I wasn't able to solve. In the end, after seeing the [first place solution](https://www.kaggle.com/c/deepfake-detection-challenge/discussion/145721), I understood that I kept a too intricate approach to the network. I could have kept it simpler, in order to concentrate more on the pre-training tasks.