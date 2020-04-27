import asyncio
import os
from asyncio.queues import Queue

import numpy as np
import pandas as pd
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from torch.utils.data import DataLoader
from tqdm import tqdm

from video_dataset import VideoDataset, collate_fn

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'


# Yield successive n-sized chunks from l
def list_chunks(l, n):
    # Looping till length l
    for i in range(0, len(l), n):
        yield l[i:i + n]


labels_dict = {'FAKE': 1, 'REAL': 0}


def extract_faces_and_embeddings(dataloader, face_detector, feature_extractor, dest_path):
    """
    Extract faces from frames. Each face will come be outputted as (n_frames, channels, height, width).
    This method yields a dictionary containing the filename, the faces vector and the label, FAKE or REAL.
    If there are no faces, the array remains empty. We should keep it as the same, in order to use the audio part.
    Some frames may not contain any face. In that case, we don't skip them but add an empty frame
    :param dataloader: dataloader for videos
    :param face_detector: detector of faces. It returns a list of faces, one for each frame from dataloader
    :param feature_extractor: feature extractor from faces. It returns a 512-long embedding for each face
    :param dest_path: path/to/folder where to save the embeddings, one for each video.
    :return:
    """
    # Define how many frames are going to be stored in GPU
    mtcnn_gpu_dim = 8
    incept_gpu_dim = 128

    for batch in tqdm(dataloader):
        for video_path, video_frame, label in zip(batch.video_path, batch.frame, batch.label):
            faces = []
            # Chunk frame list to fill them into the GPU
            for chunk in list_chunks(video_frame, mtcnn_gpu_dim):
                # TODO: remember that this method is not working until pull request from facenet-pytorch is merged
                # Change `chunk` with `[chunk[i] for i in range(chunk.shape[0]]`,it will be processed but less optimized
                faces.extend([a if a is not None else torch.zeros((3, 160, 160)) for a in
                              face_detector(chunk)])  # Do not skip None faces
            # Torch likes (..., channels, h, w) so we keep these dimensions
            # Empty CUDA cache to let the feature extractor perform well.
            torch.cuda.empty_cache()
            all_faces_embeddings = []
            if not faces:
                # Add one empty frame
                all_faces_embeddings = np.zeros((1, 512), 'float32')
            else:
                # This method will process `number of faces per frame` at the same time,
                # and store at the same embedding position
                for chunk in list_chunks(faces, incept_gpu_dim):
                    all_faces_embeddings.extend(feature_extractor(
                        torch.stack(chunk, dim=0).to(DEVICE)).detach().cpu().numpy())
            # Convert label into 1 if FAKE and 0 if REAL
            label = labels_dict[label]
            # Save video embeddings to single file
            df = pd.DataFrame(columns=['filename', 'video_embedding', 'label'])
            filename = os.path.basename(video_path)
            path = os.path.join(dest_path, filename.split('.')[0] + '.csv')
            df.loc[0] = [filename, all_faces_embeddings, label]
            df.to_pickle(path)
            # Free up CUDA memory after work is done.
            torch.cuda.empty_cache()


if __name__ == '__main__':
    dest_path = 'dataset\\video_embeddings'
    dataset = VideoDataset('data\\train_data\\to_add.json', 2, check_path=dest_path)
    dataloader = DataLoader(
        dataset,
        # Keep batch size always > 1 since the custom collate function
        # will skip None videos and it will fall back to the rest
        batch_size=6,
        # sampler=Subset[0, 1, 2],
        num_workers=3,
        pin_memory=True,
        collate_fn=collate_fn
    )
    # Load Face detector
    face_detector = MTCNN(margin=14, device=DEVICE).eval()
    # Load facial recognition model
    feature_extractor = InceptionResnetV1(pretrained='vggface2', device=DEVICE).eval()
    # Extract faces and embeddings
    extract_faces_and_embeddings(dataloader, face_detector, feature_extractor, dest_path)
