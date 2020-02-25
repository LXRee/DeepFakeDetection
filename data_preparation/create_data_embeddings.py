import torch.multiprocessing as tmp
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
# # list of paths in metadata for which there is no corresponding video
# skip_paths = [
#     'data\\dfdc_train_part_18\\wipjitfmta.mp4',
#     'data\\dfdc_train_part_18\\wpuxmawbkj.mp4',
#     'data\\dfdc_train_part_18\\pvohowzowy.mp4'
# ]


# Source: https://www.kaggle.com/timesler/facial-recognition-model-in-pytorch
def process_faces(faces, feature_extractor, device='cuda:0'):
    # Filter out frames without faces
    faces = [f for f in faces if f is not None]
    if len(faces) == 0:
        return None
    faces = torch.cat(faces).to(device)

    # Generate facial feature vectors using a pretrained model
    embeddings = feature_extractor(faces)

    # Calculate centroid for video and distance of each face's feature vector from centroid
    # centroid = embeddings.mean(dim=0)
    # x = (embeddings - centroid).norm(dim=1).cpu().numpy()

    return embeddings.detach().cpu().numpy()
    # return x


def create_images_embedding():
    # TRAIN_DIR = '/kaggle/input/deepfake-detection-challenge/train_sample_videos/'
    TRAIN_DIR = 'data'
    OUTPUT_DIR = 'embeddings'

    metadata = json.load(open(os.path.join(TRAIN_DIR, 'data.json'), 'r'))

    # define face search hyperparameters
    SCALE = 0.5
    N_FRAMES = None
    WINDOW = 6  # define how many frames skip per video
    # THREADS = 2

    # define multiprocessing work
    # pool = tmp.Pool(processes=THREADS)
    # arguments = chunks(metadata, len(metadata.keys()) // THREADS)
    # result = [pool.apply_async(do_the_job, args=(i, a)) for i, a in enumerate(arguments)]
    #  _ = [p.get() for p in result]
    # Get the paths of all train videos
    all_train_videos = list(metadata.keys())
    # Load face detector
    face_detector = MTCNN(margin=14, keep_all=True, factor=0.5, device=device).eval()

    # Load facial recognition model
    feature_extractor = InceptionResnetV1(pretrained='vggface2', device=device).eval()

    # Define face detection pipeline
    detection_pipeline = DetectionPipeline(detector=face_detector, n_frames=N_FRAMES, resize=SCALE, window=WINDOW)

    df = pd.DataFrame(columns=['filename', 'embedding', 'label'])

    j = 0
    df_part = pd.DataFrame(columns=['filename', 'embedding', 'label'])
    part_len = 1000
    first_round = True
    # keep trace of skipped video just for curiosity
    skipped_videos = 0
    with torch.no_grad():
        for path in tqdm(all_train_videos):
            if j < 51000 or j > 75000:
                j += 1
                # flag = False
            else:
                if j % part_len == 0 and not first_round:
                    print("Job completed {} operations and saved a df partition".format(j))
                    df_part.to_pickle(os.path.join(OUTPUT_DIR, 'partials', 'train_embeddings_{}.csv'.format(j // part_len)))
                    df_part = pd.DataFrame(columns=['filename', 'embedding', 'label'])
                first_round = False
                file_name = path

                try:
                    # Detect all faces occur in the video
                    faces = detection_pipeline(path)

                    # Calculate the distances of all faces' feature vectors to the centroid
                    embeddings = process_faces(faces, feature_extractor, device)
                    if embeddings is None:
                        continue
                    row = [
                        file_name,
                        embeddings,
                        1 if metadata[path]['label'] == 'FAKE' else 0
                    ]
                    # Append a new row at the end of the data frame
                    df.loc[j] = row
                    df_part.loc[j] = row

                    # del faces
                    # del embeddings
                    torch.cuda.empty_cache()
                    j += 1
                except NoFrames:
                    skipped_videos += 1

    # print(df.head())
    df.to_pickle(os.path.join(OUTPUT_DIR, 'train_embeddings_complete.csv'))
    print("Job finished after {} video processed\nSkipped {} videos".format(j, skipped_videos))


if __name__ == '__main__':
    create_images_embedding()
