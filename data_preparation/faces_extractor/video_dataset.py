import os
import json
import cv2

from custom_exceptions import NoFrames, NoFaces, NoVideo
import numpy as np
from torch.utils.data import Dataset, DataLoader


class VideoDataset(Dataset):
    def __init__(self, metadata_path, window=1, resize=0.5, transform=None):
        """
        Dataset to load videos and extract faces. I hope this class will make the process faster.
        This class will always take all the possible frames from each video.

        :param metadata_path: path to metadata file. It contains the path to all video files
        :param window: defines how many frames to skip between one take and another. 1: 0 frame skip, 2: 1 frame skipped
        :param resize: percentile of resize for frames. Too big frames lead to OOM
        :param transform: transforms for dataset
        """
        # Load metadata file
        self.metadata = json.load(open(os.path.join(metadata_path, 'metadata.json'), 'r'))
        # Define path to all videos to access them with getitem
        self.path_to_all_videos = list(self.metadata.keys())
        # Define window: steps for reading video
        self.window = window
        # Define resize measure for faces. The feature extractor has been pre-trained with 160x160 images, so we are
        # going to keep this dimension
        self.resize = resize
        # Define transforms for Dataset
        self.transform = transform
        # Define list of error path to retrieve later if needed
        self.error_paths = []

    def __len__(self):
        return len(self.path_to_all_videos)

    def __getitem__(self, idx):
        try:
            # Create video reader and find length
            video_path = self.path_to_all_videos[idx]

            v_cap = cv2.VideoCapture(video_path)
            v_len = int(v_cap.get(cv2.CAP_PROP_FRAME_COUNT))

            # In case of missing video file just skip and raise exception
            if v_len == 0:
                raise NoVideo(video_path)

            # Define frame height and width to prepare an empty numpy vector
            v_height = int(v_cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * self.resize)
            v_width = int(v_cap.get(cv2.CAP_PROP_FRAME_WIDTH) * self.resize)

            # Prepare emtpy batch of frames
            # frames = np.zeros((batch_size, v_height, v_width, 3), dtype='uint8')
            frames = []

            # Actually extract frames
            for i in range(v_len):
                # Select next frame
                _ = v_cap.grab()
                if i % self.window == 0:
                    # Load frame
                    success, frame = v_cap.retrieve()

                    # Skip frame if retrieve is not successful
                    if not success:
                        continue

                    # Decode and resize frame
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame = cv2.resize(frame, (v_width, v_height))

                    # Put frame into batch of frames
                    frames.append(frame)
            if frames:
                return {
                    'video_path': video_path,
                    'frame': np.array(frames, dtype='uint8'),
                    'label': self.metadata[video_path]['label']
                }
            # If no frame has been retrieved, raise exception
            else:
                raise NoFrames(video_path)

        # Manage exceptions
        except NoVideo as e:
            print("Video {} not found".format(e))
            self.error_paths.append(e)
            return None
        except NoFrames as e:
            print("Video {} has no frames".format(e))
            self.error_paths.append(e)
            return None


def collate_fn(batch):
    """
    Custom collate function to skip None videos and to manage multiple resolution videos from source
    :param batch:
    :return:
    """
    # Filter out None videos (that is, the ones that triggered some custom_exceptions)
    batch = list(filter(lambda x: x is not None, batch))
    # Create lists for batch
    video_paths = []
    frames = []
    labels = []
    for el in batch:
        video_paths.append(el['video_path'])
        frames.append(el['frame'])
        labels.append(el['label'])
    return {
        'video_path': video_paths,
        'frame': frames,
        'label': labels
    }


if __name__ == '__main__':
    from PIL import Image
    dataset = VideoDataset('C:\\Users\\mawanda\\PyCharmProjects\\DeepFakeCompetition\\data\\train_data')
    dataloader = DataLoader(
        dataset,
        batch_size=2,
        # sampler=Subset[0, 1, 2],
        num_workers=0,
        pin_memory=True
    )
    for batch in dataloader:
        video_paths = batch['video_path']  # List of batch_size paths
        video_frames = batch['frame']  # List of frames: (batch_size, n_frames, height, width, channels)
        labels = batch['label']  # List of labels: FAKE, REAL
        # proof_frame = Image.fromarray(video_frames[0][0].detach().cpu().numpy()).show()
        # print("Video_paths: {}".format(video_paths))
        # print("Video frames: {}".format(video_frames))
        # print("Video label: {}".format(labels))



