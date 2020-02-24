import os
import json
import torch

import pandas as pd
from tqdm import tqdm
from facenet_pytorch import MTCNN, InceptionResnetV1
from source.dataset.DetectionPipeline import DetectionPipeline
from source.helper.faces import process_faces

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(f'Running on device: {device}')


def main():
    # TRAIN_DIR = '/kaggle/input/deepfake-detection-challenge/train_sample_videos/'
    TRAIN_DIR = 'data'


    BATCH_SIZE = 1
    SCALE = 0.25
    N_FRAMES = 10

    # Load face detector
    face_detector = MTCNN(margin=14, keep_all=True, factor=0.5, device=device).eval()

    # Load facial recognition model
    feature_extractor = InceptionResnetV1(pretrained='vggface2', device=device).eval()

    # Define face detection pipeline
    detection_pipeline = DetectionPipeline(detector=face_detector, n_frames=N_FRAMES, batch_size=BATCH_SIZE, resize=SCALE)

    # Get the paths of all train videos
    metadata = json.load(open(os.path.join(TRAIN_DIR, 'data.json'), 'r'))
    # all_train_videos = glob.glob(os.path.join(TRAIN_DIR, '*.mp4'))
    all_train_videos = list(metadata.keys())

    df = pd.DataFrame(columns=['filename', 'distance', 'label'])

    with torch.no_grad():
        for path in tqdm(all_train_videos):
            file_name = path

            # Detect all faces occur in the video
            faces = detection_pipeline(path)

            # Calculate the distances of all faces' feature vectors to the centroid
            distances = process_faces(faces, feature_extractor)
            if distances is None:
                continue

            for distance in distances:
                row = [
                    file_name,
                    distance,
                    1 if metadata[path]['label'] == 'FAKE' else 0
                ]

                # Append a new row at the end of the data frame
                df.loc[len(df)] = row

    print(df.head())
    df.to_csv('train.csv', index=False)


if __name__ == '__main__':
    main()
