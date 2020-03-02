import json
import os

import pandas as pd
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from tqdm import tqdm

from data_preparation.embeddings.classes.FaceDetectionPipeline import FaceDetectionPipeline
from data_preparation.utils.custom_exceptions import NoFrames

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
# device = 'cpu'
print(f'Running on device: {device}')


# Source: https://www.kaggle.com/timesler/facial-recognition-model-in-pytorch

def process_faces(faces, feature_extractor, device='cuda:0'):
    # Filter out frames without faces
    faces = [f for f in faces if f is not None]
    if len(faces) == 0:
        return None

    # Send faces array to device
    faces = torch.cat(faces).to(device)

    # Generate facial feature vectors using a pre-trained model.
    # Keep the generated features (512 per face) as an "embedding" vector to be fed into a LSTM network.
    # My guess is that recognize fake faces is simpler if they are moving.
    embeddings = feature_extractor(faces)

    return embeddings.detach().cpu().numpy()


def create_submission_embedding():
    # TRAIN_DIR = '/kaggle/input/deepfake-detection-challenge/train_sample_videos/'
    TEST_DIR = 'test_data'
    OUTPUT_DIR = ''

    # scale image for computation. I'm keeping it as big as possible to facilitate feature extraction
    SCALE = 0.5
    # number of frames to keep from the video. None will keep as many as possible
    N_FRAMES = None
    # define how many frame to skip from one another
    WINDOW = 6

    # Get the paths of all train videos
    all_test_videos = []
    for root, dirs, files in os.walk(TEST_DIR):
        all_test_videos.extend([os.path.join(root, file) for file in files])

    # Load face detector
    face_detector = MTCNN(margin=14, keep_all=True, factor=0.5, device=device).eval()

    # Load facial recognition model
    feature_extractor = InceptionResnetV1(pretrained='vggface2', device=device).eval()

    # Define face detection pipeline
    detection_pipeline = FaceDetectionPipeline(detector=face_detector, n_frames=N_FRAMES, resize=SCALE, window=WINDOW)

    # Create empty dataframe that will receive faces embeddings
    df = pd.DataFrame(columns=['filename', 'embedding'])

    # Keep trace of skipped video (no video, but present in metadata) just for curiosity
    skipped_videos = 0

    j = 0
    with torch.no_grad():
        for path in tqdm(all_test_videos):
            try:
                # Detect all faces occur in the video
                faces = detection_pipeline(path)

                # Calculate faces' features embeddings
                embeddings = process_faces(faces, feature_extractor, device)

                if embeddings is None:
                    continue

                # Define row to save in dataframe
                row = [path, embeddings]

                # Append a new row at the end of the data frame
                df.loc[j] = row

                j += 1
            except NoFrames:
                # The path points to a missing video. Just skip it.
                skipped_videos += 1

    # Save entire dataframe.
    # Most of the time is useless, since we have to manage too many videos and partitions are more feasible
    df.to_pickle(os.path.join(OUTPUT_DIR, 'test_video_embeddings.csv'))
    print("Job finished after {} video processed\nSkipped {} videos".format(j, skipped_videos))


def create_images_embedding():
    # TODO: pass to function
    TRAIN_DIR = 'test_data'
    OUTPUT_DIR = ''

    # Load metadata.
    # Metadata is a dictionary with key: path, value: label.
    # It has also other keys, but I'm not interested in them. Just go explore in case.
    metadata = json.load(open(os.path.join(TRAIN_DIR, 'metadata.json'), 'r'))

    # Scale image for computation. I'm keeping it as big as possible to facilitate feature extraction
    SCALE = 0.5
    # Number of frames to keep from the video. None will keep as many as possible
    N_FRAMES = None
    # Define how many frame to skip from one another
    WINDOW = 6

    # Get the paths of all train videos
    all_train_videos = list(metadata.keys())

    # Load face detector
    face_detector = MTCNN(margin=14, keep_all=True, factor=0.5, device=device).eval()

    # Load facial recognition model
    feature_extractor = InceptionResnetV1(pretrained='vggface2', device=device).eval()

    # Define face detection pipeline
    detection_pipeline = FaceDetectionPipeline(detector=face_detector, n_frames=N_FRAMES, resize=SCALE, window=WINDOW)

    # Create empty dataframe that will receive faces embeddings
    df = pd.DataFrame(columns=['filename', 'embedding', 'label'])

    # Create empty partition dataframe to store faces embeddings progressively.
    df_part = pd.DataFrame(columns=['filename', 'embedding', 'label'])

    # Index to decide which frames to control
    j = 0

    # Number of videos to keep per partition dataframe
    part_len = 1000000

    # Useful flag to skip saving partition if first round
    first_round = True

    # Keep trace of skipped video (no video, but present in metadata) just for curiosity
    skipped_videos = 0

    # No gradient updates for networks
    with torch.no_grad():
        for name in tqdm(all_train_videos):
            path = os.path.join(TRAIN_DIR, name)
            # Decide which videos to skip
            if j < 0:
                j += 1
            else:
                # Save partition
                if j % part_len == 0 and not first_round:
                    print("\nJob completed {}->{} operations and saved a df partition.\n{} videos skipped.\n"
                          .format(j - part_len, j, skipped_videos))
                    df_part.to_pickle(os.path.join(OUTPUT_DIR,
                                                   'partials',
                                                   'train_embeddings_{}.csv'.format(j // part_len)))
                    df_part = pd.DataFrame(columns=['filename', 'embedding', 'label'])
                first_round = False

                try:
                    # Detect all faces occur in the video
                    faces = detection_pipeline(path)

                    # Calculate features embeddings of faces
                    embeddings = process_faces(faces, feature_extractor, device)

                    if embeddings is None:
                        continue

                    # Define row to save in dataframe
                    row = [path, embeddings, 1 if metadata[name]['label'] == 'FAKE' else 0]

                    # Append a new row at the end of the data frame
                    df.loc[j] = row
                    df_part.loc[j] = row

                    j += 1
                except NoFrames:
                    # The path points to a missing video. Just skip it.
                    skipped_videos += 1
                    if j % part_len == 0:
                        first_round = True

    # Save entire dataframe.
    # Most of the time is useless, since we have to manage too many videos and partitions are more feasible
    # TODO: pass to function
    df.to_pickle(os.path.join(OUTPUT_DIR, 'test_video_embeddings.csv'))
    print("Job finished after {} video processed\nSkipped {} videos".format(j, skipped_videos))


if __name__ == '__main__':
    create_images_embedding()
    # create_test_embedding()
