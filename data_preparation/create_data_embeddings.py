import os
import json
import torch
import pandas as pd
from tqdm import tqdm
from facenet_pytorch import MTCNN, InceptionResnetV1

from source.data_preparation.DetectionPipeline import DetectionPipeline
from source.data_preparation.helper.custom_exceptions import NoFrames

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
# device = 'cpu'
print(f'Running on device: {device}')


# Source: https://www.kaggle.com/timesler/facial-recognition-model-in-pytorch
def process_faces(faces, feature_extractor, device='cuda:0'):
    # Filter out frames without faces
    faces = [f for f in faces if f is not None]
    if len(faces) == 0:
        return None

    # send faces array to device
    faces = torch.cat(faces).to(device)

    # Generate facial feature vectors using a pretrained model.
    # Keep the generated features (512 per face) as an "embedding" vector to be fed into a LSTM network.
    # My guess is that recognize fake faces is simpler if they are moving.
    embeddings = feature_extractor(faces)

    return embeddings.detach().cpu().numpy()


def create_images_embedding():
    # TRAIN_DIR = '/kaggle/input/deepfake-detection-challenge/train_sample_videos/'
    TRAIN_DIR = 'data'
    OUTPUT_DIR = 'embeddings'

    # load metadata.
    # Metadata is a dictionary with key: path, value: label.
    # It has also other keys, but I'm not interested in them. Just go explore in case.
    metadata = json.load(open(os.path.join(TRAIN_DIR, 'data.json'), 'r'))

    # scale image for computation. I'm keeping it as big as possible to facilitate feature extraction
    SCALE = 0.5
    # number of frames to keep from the video. None will keep as many as possible
    N_FRAMES = None
    # define how many frame to skip from one another
    WINDOW = 6

    # Get the paths of all train videos
    all_train_videos = list(metadata.keys())

    # Load face detector
    face_detector = MTCNN(margin=14, keep_all=True, factor=0.5, device=device).eval()

    # Load facial recognition model
    feature_extractor = InceptionResnetV1(pretrained='vggface2', device=device).eval()

    # Define face detection pipeline
    detection_pipeline = DetectionPipeline(detector=face_detector, n_frames=N_FRAMES, resize=SCALE, window=WINDOW)

    # create empty dataframe that will receive faces embeddings
    df = pd.DataFrame(columns=['filename', 'embedding', 'label'])
    # create empty partition dataframe to store faces embeddings progressively.
    df_part = pd.DataFrame(columns=['filename', 'embedding', 'label'])

    # index to decide which frames to control
    j = 0

    # number of videos to keep per partition dataframe
    part_len = 1000

    # useful flag to skip saving partition if first round
    first_round = True

    # keep trace of skipped video (no video, but present in metadata) just for curiosity
    skipped_videos = 0

    # no gradient updates for networks
    with torch.no_grad():
        for path in tqdm(all_train_videos):
            # decide which videos to skip
            if j < 76000 or j > 100000:
                j += 1
            else:
                # save partition
                if j % part_len == 0 and not first_round:
                    print("\nJob completed {}->{} operations and saved a df partition.\n{} videos skipped.\n".format(j-part_len, j, skipped_videos))
                    df_part.to_pickle(os.path.join(OUTPUT_DIR, 'partials', 'train_embeddings_{}.csv'.format(j // part_len)))
                    df_part = pd.DataFrame(columns=['filename', 'embedding', 'label'])
                first_round = False

                try:
                    # Detect all faces occur in the video
                    faces = detection_pipeline(path)

                    # Calculate faces' features embeddings
                    embeddings = process_faces(faces, feature_extractor, device)

                    if embeddings is None:
                        continue
                    # define row to save in dataframe
                    row = [
                        path,
                        embeddings,
                        1 if metadata[path]['label'] == 'FAKE' else 0
                    ]

                    # Append a new row at the end of the data frame
                    df.loc[j] = row
                    df_part.loc[j] = row

                    j += 1
                except NoFrames:
                    # the path points to a missing video. Just skip it.
                    skipped_videos += 1
                    if j % part_len == 0:
                        first_round = True

    # save entire dataframe. Most of the time is useless, since we have to manage to many, many videos and partitions
    # are more feasible.
    df.to_pickle(os.path.join(OUTPUT_DIR, 'train_embeddings_complete.csv'))
    print("Job finished after {} video processed\nSkipped {} videos".format(j, skipped_videos))


if __name__ == '__main__':
    create_images_embedding()
