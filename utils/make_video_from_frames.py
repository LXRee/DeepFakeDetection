import cv2
import numpy as np
import os
from tqdm import tqdm


def video_from_frames(folder, filepath):
    """
    Convert each folder with frames into a video
    :param folder: path/to/folder where there are the frames
    :param filepath: path/to/file where to save frames
    :return: None
    """
    img_array = []
    size = None
    n_video = folder.split(os.sep)[-1]
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            height, width, layers = img.shape
            size = (width, height)
            img_array.append(img)
    if img_array:
        filepath = filepath.split('.')[0] + '_' + str(n_video) + '.' + filepath.split('.')[1]
        out = cv2.VideoWriter(filepath, 0x00000021, 15.0, size)
        for frame in tqdm(img_array, desc='{}'.format(os.path.basename(filepath))):
            out.write(frame)
        out.release()
    else:
        print("No frames in path {}".format(folder))


def read_folders(source_path):
    """
    Reads all the folders and return a list containing a list of path/to/folders where
    there are only frames - no sub folders
    :param source_path: path/to/source/path
    :return: list of paths/to/folders with only frames inside
    """
    folders = set()
    for root, dirs, files in tqdm(os.walk(source_path), desc="Reading folders from source..."):
        for _ in files:
            folders.add(root)
    return folders


if __name__ == '__main__':
    folders = read_folders(os.path.join('data', 'youtube_faces'))
    dest_path = os.path.join('data', 'youtube_faces_video')
    for folder in folders:
        filepath = os.path.join(dest_path, folder.split(os.sep)[-2]) + '.mp4'
        video_from_frames(folder, filepath)
