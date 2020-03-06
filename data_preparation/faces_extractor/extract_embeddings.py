import torch
from torch.utils.data import DataLoader
from video_dataset import VideoDataset, collate_fn
import pandas as pd
from tqdm import tqdm
import numpy as np
import os
from facenet_pytorch import MTCNN, InceptionResnetV1
from asyncio.queues import Queue
import asyncio

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'


# Yield successive n-sized
# chunks from l.
def list_chunks(l, n):
    # looping till length l
    for i in range(0, len(l), n):
        yield l[i:i + n]


labels_dict = {'FAKE': 1, 'REAL': 0}


async def extract_faces(dataloader):
    """
    Extract faces from frames. Each face will come be outputted as (n_frames, channels, height, width).
    This method yields a dictionary containing the filename, the faces vector and the label, FAKE or REAL.
    If there are no faces, the array remains empty. We should keep it as the same, in order to use the audio part.
    Some frames may not contain any face. In that case, we don't skip them but add an empty frame
    :param dataloader: dataloader for videos
    :return:
    """
    # Load face detector
    face_detector = MTCNN(margin=14, device=DEVICE).eval()
    # Define how many frames are going to be stored in GPU
    gpu_dim = 30

    for batch in tqdm(dataloader):
        for video_path, video_frame, label in zip(batch['video_path'], batch['frame'], batch['label']):
            faces = []
            # Chunk frame list to fill them into the GPU
            for chunk in list_chunks(video_frame, gpu_dim):
                # TODO: remember that this method is not working until pull request from facenet-pytorch is merged
                # Change `chunk` with `[chunk[i] for i in range(chunk.shape[0]]`,it will be processed but less optimized
                faces.extend([a if a is not None else torch.zeros((3, 160, 160)) for a in face_detector(chunk)])  # Do not skip None faces
            # There may be more than one face in video. So we keep both in the first dimension
            # Torch likes (..., channels, h, w) so we keep these dimensions
            filename = os.path.basename(video_path)
            label = label

            yield {'filename': filename, 'faces': faces, 'label': label}
            torch.cuda.empty_cache()


async def extract_faces_features(queue, feature_extractor, dest_path):
    """
    Extract features from faces crops. Retrieve asynchronously the features from each face
    :param queue: queue of faces
    :return:
    """
    while True:
        # consume faces
        info_faces = await queue.get()
        if info_faces is not None:
            filename = info_faces['filename']
            # Remember that faces = list(n_frames) with (n_faces, height, width, channels), therefore we are going
            # to create a new embedding for each face.
            faces_list = info_faces['faces']
            if not faces_list:
                # Add one empty frame
                all_faces_embeddings = np.zeros((1, 512), 'float32')
            else:
                # This method will process `number of faces per frame` at the same time, and store at the same embedding position
                all_faces_embeddings = feature_extractor(torch.stack(faces_list, dim=0).to(DEVICE)).detach().cpu().numpy()
            # Convert label into 1 if FAKE and 0 if REAL
            label = labels_dict[info_faces['label']]
            # Save video embeddings to single file
            df = pd.DataFrame(columns=['filename', 'video_embedding', 'label'])
            df.loc[0] = [filename, all_faces_embeddings, label]
            path = os.path.join(dest_path, filename.split('.')[0] + '.csv')
            df.to_pickle(path)
            # Notify when the "work item" has been processed
            queue.task_done()
        else:
            break


async def data_producer(queue, dataloader):
    """
    Create queue - not too big not to overload memory
    Put data into queue
    """
    async for d in extract_faces(dataloader):
        await queue.put(d)
    await queue.put(None)


if __name__ == '__main__':
    dataset = VideoDataset('C:\\Users\\mawanda\\PyCharmProjects\\DeepFakeCompetition\\data\\train_data', 4)
    # dataset = VideoDataset('C:\\Users\\mawanda\\Desktop\\prova', 6)
    dest_path = 'data/train_faces'
    dataloader = DataLoader(
        dataset,
        # Keep batch size always > 1 since the custom collate function will skip None videos and it will fall back to the rest.
        batch_size=6,
        # sampler=Subset[0, 1, 2],
        num_workers=2,
        pin_memory=True,
        collate_fn=collate_fn
    )
    # extract_faces(dataloader)

    # Load facial recognition model
    feature_extractor = InceptionResnetV1(pretrained='vggface2', device=DEVICE).eval()

    # Create async queue to process videos and embeddings concurrently
    # Reference: https://asyncio.readthedocs.io/en/latest/producer_consumer.html
    loop = asyncio.get_event_loop()
    queue = Queue(10, loop=loop)
    producer_coro = data_producer(queue, dataloader)
    consumer_coro = extract_faces_features(queue, feature_extractor, dest_path)

    loop.run_until_complete(asyncio.gather(producer_coro, consumer_coro))
    loop.close()


